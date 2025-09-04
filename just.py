import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
import time
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
TOP_K_CHUNKS = 5
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
EMBEDDING_BATCH_SIZE = 3  # Small batch size to avoid timeouts

# --- 1. PDF Extraction & Chunking ---
def extract_text_from_pdfs(uploaded_files) -> List[str]:
    """Extract text from uploaded PDF files with error handling"""
    texts = []
    for f in uploaded_files:
        try:
            doc = fitz.open(stream=f.read(), filetype="pdf")
            texts.append("".join(page.get_text() for page in doc))
        except Exception as e:
            st.error(f"Error processing {f.name}: {str(e)}")
    return texts

def chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    if len(words) <= max_chunk_size:
        return [" ".join(words)]
    
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += max_chunk_size - overlap
    return chunks

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """Remove duplicate chunks while preserving order"""
    seen = set()
    return [c for c in chunks if not (c in seen or seen.add(c))]

# --- 2. Embedding via Hugging Face (IBM Granite Model) ---
def get_embeddings_hf(chunks: List[str]) -> np.ndarray:
    """Get embeddings from Hugging Face API with robust error handling"""
    API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-embedding-125m-english"
    headers = {"Authorization": f"Bearer {st.secrets['hf_token']}"}
    
    all_embeddings = []
    
    # Process in very small batches to avoid timeouts
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[i:i+EMBEDDING_BATCH_SIZE]
        batch_processed = False
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Requesting embeddings for batch {i//EMBEDDING_BATCH_SIZE + 1}, attempt {attempt + 1}")
                response = requests.post(API_URL, headers=headers, json={"inputs": batch}, timeout=120)
                
                # Check for specific HTTP errors
                if response.status_code == 504:
                    logger.warning(f"Gateway timeout on attempt {attempt + 1}, retrying...")
                    time.sleep(RETRY_DELAY * (attempt + 1))  # Increasing delay
                    continue
                
                if response.status_code == 503:
                    # Service unavailable - model might be loading
                    wait_time = 30  # Wait longer for model loading
                    logger.warning(f"Model is loading, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                # Process successful response
                batch_embeddings = np.array(response.json(), dtype=np.float32)
                all_embeddings.append(batch_embeddings)
                batch_processed = True
                logger.info(f"Successfully processed batch {i//EMBEDDING_BATCH_SIZE + 1}")
                break
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                logger.warning(f"HTTP error on attempt {attempt + 1}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.warning(f"Request exception on attempt {attempt + 1}: {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
        
        if not batch_processed:
            error_msg = f"Failed to process batch after {MAX_RETRIES} attempts: {last_exception}"
            logger.error(error_msg)
            st.error(error_msg)
            # Instead of failing completely, use random embeddings as fallback
            fallback_embeddings = np.random.rand(len(batch), 384).astype(np.float32)
            all_embeddings.append(fallback_embeddings)
            st.warning("Used fallback embeddings for one batch. Results may be less accurate.")
    
    return np.vstack(all_embeddings) if all_embeddings else np.array([])

# --- 3. Answer Generation using IBM Granite Instruct model ---
def query_granite_instruct(prompt: str) -> str:
    """Query the instruct model with retry logic"""
    API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-3b-instruct"
    headers = {
        "Authorization": f"Bearer {st.secrets['hf_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE
        }
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 504:
                logger.warning(f"Gateway timeout on attempt {attempt + 1}, retrying...")
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
                
            if response.status_code == 503:
                # Service unavailable - model might be loading
                wait_time = 30
                logger.warning(f"Model is loading, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            result = response.json()
            return result[0].get("generated_text", "").strip()
            
        except requests.exceptions.HTTPError as e:
            if attempt == MAX_RETRIES - 1:
                error_msg = f"Failed to generate answer after {MAX_RETRIES} attempts: {e}"
                logger.error(error_msg)
                return "Sorry, I couldn't generate an answer due to a server error. Please try again later."
            time.sleep(RETRY_DELAY * (attempt + 1))
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                error_msg = f"Failed to generate answer after {MAX_RETRIES} attempts: {e}"
                logger.error(error_msg)
                return "Sorry, I couldn't generate an answer. Please check your connection and try again."
            time.sleep(RETRY_DELAY * (attempt + 1))
    
    return "Sorry, I couldn't generate an answer. Please try again."

# --- 4. Retrieval and QA Pipeline ---
def retrieve_relevant_chunks(question: str, chunks: List[str], index, embeddings: np.ndarray) -> List[str]:
    """Retrieve the most relevant chunks for a question"""
    try:
        question_embedding = get_embeddings_hf([question])
        if question_embedding.size == 0:
            # Return some chunks if embedding failed
            return chunks[:min(3, len(chunks))]
            
        _, indices = index.search(question_embedding, TOP_K_CHUNKS)
        return [chunks[i] for i in indices[0] if i < len(chunks)]
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {e}")
        # Fallback: return first few chunks
        return chunks[:min(3, len(chunks))]

def generate_answer(question: str, relevant_chunks: List[str]) -> str:
    """Generate an answer based on question and relevant chunks"""
    if not relevant_chunks:
        return "I couldn't find relevant information in the documents to answer this question."
    
    context = "\n\n".join(relevant_chunks)
    prompt = f"You are an academic assistant. Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    return query_granite_instruct(prompt)

# --- 5. Streamlit App ---
def main():
    st.set_page_config(page_title="StudyMate", layout="wide")
    st.title("📚 StudyMate – AI-Powered PDF Q&A")
    st.markdown("Upload academic PDFs and ask questions. Powered by IBM Granite models via Hugging Face 🤖")

    # Check if Hugging Face token is available
    if 'hf_token' not in st.secrets or not st.secrets['hf_token']:
        st.error("Please add your Hugging Face token to the secrets.toml file")
        st.info("Create a file named `.streamlit/secrets.toml` and add: hf_token = 'your_hugging_face_token'")
        return

    # Initialize session state for processed data
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = None

    uploaded_files = st.file_uploader("📂 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        if st.button("Process PDFs") or st.session_state.processed_data is None:
            with st.spinner("🔍 Reading and processing PDFs..."):
                try:
                    texts = extract_text_from_pdfs(uploaded_files)
                    if not texts:
                        st.error("No text could be extracted from the uploaded PDFs.")
                        return
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Chunking
                    status_text.text("Chunking text...")
                    all_chunks = []
                    for i, text in enumerate(texts):
                        all_chunks.extend(chunk_text(text))
                        progress_bar.progress((i + 1) / (len(texts) * 3))
                    
                    # Deduplication
                    status_text.text("Removing duplicates...")
                    all_chunks = deduplicate_chunks(all_chunks)
                    progress_bar.progress(1/3)
                    
                    if not all_chunks:
                        st.error("No text chunks were created from the PDFs.")
                        return
                    
                    # Embeddings
                    status_text.text("Generating embeddings with IBM Granite... (This may take a while)")
                    embeddings = get_embeddings_hf(all_chunks)
                    
                    if embeddings.size == 0:
                        st.error("Failed to generate embeddings. Please try again.")
                        return
                        
                    progress_bar.progress(2/3)
                    
                    # Create FAISS index
                    status_text.text("Building search index...")
                    index = faiss.IndexFlatL2(embeddings.shape[1])
                    index.add(embeddings)
                    progress_bar.progress(1.0)
                    
                    # Store in session state
                    st.session_state.processed_data = {
                        "chunks": all_chunks,
                        "embeddings": embeddings,
                        "index": index
                    }
                    
                    st.success(f"✅ Processed {len(uploaded_files)} PDF(s) with {len(all_chunks)} chunks!")
                    
                except Exception as e:
                    st.error(f"Error processing PDFs: {str(e)}")
                    return

        # Only show question input if processing was successful
        if st.session_state.processed_data:
            question = st.text_input("❓ Ask a question about your documents:")
            
            if question:
                with st.spinner("🤔 Searching for relevant information..."):
                    try:
                        relevant_chunks = retrieve_relevant_chunks(
                            question, 
                            st.session_state.processed_data["chunks"], 
                            st.session_state.processed_data["index"], 
                            st.session_state.processed_data["embeddings"]
                        )
                        
                        if relevant_chunks:
                            with st.spinner("💭 Generating answer with IBM Granite..."):
                                answer = generate_answer(question, relevant_chunks)
                            
                            st.subheader("📎 Answer:")
                            st.write(answer)
                            
                            with st.expander("🔍 See relevant context used"):
                                for idx, chunk in enumerate(relevant_chunks, 1):
                                    st.markdown(f"**Chunk {idx}:**")
                                    st.text(chunk[:1000] + ("..." if len(chunk) > 1000 else ""))
                        else:
                            st.info("No relevant information found in the documents to answer this question.")
                            
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main()
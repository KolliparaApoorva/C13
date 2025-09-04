import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import requests

# --- 1. PDF Extraction & Chunking ---
def extract_text_from_pdfs(uploaded_files):
    texts = []
    for f in uploaded_files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        texts.append("".join(page.get_text() for page in doc))
    return texts

def chunk_text(text, max_chunk_size=500, overlap=20):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + max_chunk_size]))
        start += max_chunk_size - overlap
    return chunks

def deduplicate_chunks(chunks):
    seen = set()
    return [c for c in chunks if not (c in seen or seen.add(c))]

# --- 2. Embedding via Hugging Face (IBM Granite Model) ---
def get_embeddings_hf(chunks):
    API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-embedding-125m-english"
    headers = {"Authorization": f"Bearer {st.secrets['hf_token']}"}
    payload = {"inputs": chunks}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return np.array(response.json(), dtype=np.float32)

# --- 3. Answer Generation using IBM Granite Instruct model ---
def query_granite_instruct(prompt):
    API_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-3b-instruct"
    headers = {
        "Authorization": f"Bearer {st.secrets['hf_token']}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()[0].get("generated_text", "")

# --- 4. Retrieval and QA Pipeline ---
def retrieve_relevant_chunks(question, chunks, index, embeddings):
    question_embedding = get_embeddings_hf([question])
    _, indices = index.search(question_embedding, 5)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_answer(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = f"You are an academic assistant.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    return query_granite_instruct(prompt).strip()

# --- 5. Streamlit App ---
def main():
    st.set_page_config(page_title="StudyMate", layout="wide")
    st.title("📚 StudyMate – AI-Powered PDF Q&A")
    st.markdown("Upload academic PDFs and ask questions. Powered by IBM Watsonx via Hugging Face 🤖")

    uploaded_files = st.file_uploader("📂 Upload one or more PDF files", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("🔍 Reading and processing PDFs..."):
            texts = extract_text_from_pdfs(uploaded_files)
            all_chunks = deduplicate_chunks(
                chunk for text in texts for chunk in chunk_text(text)
            )
            embeddings = get_embeddings_hf(all_chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
        st.success("✅ PDF(s) processed! You can now ask questions.")

        question = st.text_input("❓ Ask a question:")
        if question:
            with st.spinner("🤔 Thinking..."):
                relevant_chunks = retrieve_relevant_chunks(question, all_chunks, index, embeddings)
                answer = generate_answer(question, relevant_chunks)

            st.subheader("📎 Answer:")
            st.write(answer)

            with st.expander("🔍 Context Chunks Used"):
                for idx, chunk in enumerate(relevant_chunks, 1):
                    st.markdown(f"**Chunk {idx}:** {chunk[:500]}...")

if __name__ == "__main__":
    main()

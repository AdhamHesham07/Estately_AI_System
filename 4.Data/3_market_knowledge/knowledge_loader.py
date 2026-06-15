import os
import sys
import glob
import faiss
import numpy as np
import requests
import pickle
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)
load_dotenv(os.path.join(BASE_DIR, ".env"))

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    print("!!! [WARNING] GEMINI_API_KEY not found in .env. Embeddings will fail.")
    sys.exit(1)

KNOWLEDGE_DIR = os.path.dirname(__file__)
RAW_DOCS_DIR = os.path.join(KNOWLEDGE_DIR, "raw_documents")
FAISS_INDEX_DIR = os.path.join(KNOWLEDGE_DIR, "faiss_index")
os.makedirs(RAW_DOCS_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts):
    # Process texts individually since batchEmbedContent requires a different format
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={gemini_key}"
    all_embeddings = []
    
    for text in texts:
        payload = {
            "content": {"parts": [{"text": text}]}
        }
        response = requests.post(api_url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Gemini API Error: {response.text}")
            
        data = response.json()
        emb = data.get("embedding", {}).get("values")
        if not emb:
            raise Exception("No embedding found in Gemini response.")
            
        all_embeddings.append(emb)
        
    return np.array(all_embeddings, dtype=np.float32)

def ingest_knowledge():
    print(f"--- [RAG Ingestion] Scanning {RAW_DOCS_DIR} for market reports ---")
    chunks = []
    metadata = []
    
    for file_path in glob.glob(os.path.join(RAW_DOCS_DIR, "**/*.txt"), recursive=True):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                file_chunks = chunk_text(content)
                for chunk in file_chunks:
                    chunks.append(chunk)
                    metadata.append({"source": os.path.basename(file_path)})
        except Exception as e:
            print(f"[RAG Ingestion] Error reading {file_path}: {e}")
            
    if not chunks:
        print("No chunks found.")
        return
        
    print(f"--- [RAG Ingestion] Generated {len(chunks)} chunks. Generating Embeddings... ---")
    
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        emb = get_embeddings(batch)
        all_embeddings.append(emb)
        print(f"Processed batch {i//batch_size + 1}/{len(chunks)//batch_size + 1}")
        
    final_embeddings = np.vstack(all_embeddings)
    
    print("--- [RAG Ingestion] Building FAISS Vector Store... ---")
    dimension = final_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(final_embeddings)
    
    faiss.write_index(index, os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    with open(os.path.join(FAISS_INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump((chunks, metadata), f)
        
    print(f"--- [RAG Ingestion] Successfully saved FAISS index to {FAISS_INDEX_DIR} ---")

if __name__ == "__main__":
    ingest_knowledge()

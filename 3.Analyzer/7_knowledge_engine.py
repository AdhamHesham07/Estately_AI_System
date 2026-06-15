import os
import sys
import faiss
import pickle
import numpy as np
import requests
from dotenv import load_dotenv

# Ensure cross-module imports work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

class KnowledgeEngine:
    """
    RAG Retrieval System for the Analyzer.
    Loads the local FAISS index built by the Data Acquisition layer 
    and performs semantic similarity searches using direct API calls.
    """
    
    _vector_store = None
    _chunks = None
    _metadata = None

    @classmethod
    def _initialize(cls):
        if cls._vector_store is not None:
            return True
            
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("!!! [WARNING] GEMINI_API_KEY not found. RAG Knowledge Engine is offline.")
            return False
            
        try:
            faiss_index_dir = os.path.abspath(os.path.join(BASE_DIR, "4.Data", "3_market_knowledge", "faiss_index"))
            index_path = os.path.join(faiss_index_dir, "index.faiss")
            pkl_path = os.path.join(faiss_index_dir, "metadata.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(pkl_path):
                print(f"!!! [WARNING] FAISS index not found at {faiss_index_dir}. Run knowledge_loader.py first.")
                return False
                
            # Load the vector store and metadata into memory
            cls._vector_store = faiss.read_index(index_path)
            with open(pkl_path, "rb") as f:
                cls._chunks, cls._metadata = pickle.load(f)
                
            return True
        except Exception as e:
            print(f"!!! [ERROR] Failed to load RAG Vector Store: {e}")
            return False

    @staticmethod
    def _get_embedding(text: str) -> np.ndarray:
        gemini_key = os.getenv("GEMINI_API_KEY")
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={gemini_key}"
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
            
        return np.array([emb], dtype=np.float32)

    @classmethod
    def retrieve_context(cls, query: str, top_k: int = 3) -> str:
        """
        Executes a dense vector search against the market knowledge base.
        Returns a formatted string of the most relevant qualitative insights.
        """
        if not cls._initialize():
            return "No qualitative market context available at this time."
            
        try:
            query_emb = cls._get_embedding(query)
            
            # Perform similarity search
            distances, indices = cls._vector_store.search(query_emb, top_k)
            
            if len(indices[0]) == 0 or indices[0][0] == -1:
                return "No highly relevant market context found in the knowledge base."
                
            # Compile the retrieved chunks into a dense context block
            context_payload = "[RAG_MARKET_CONTEXT]\n"
            for idx_count, idx in enumerate(indices[0], start=1):
                if idx == -1:
                    continue
                content = cls._chunks[idx]
                meta = cls._metadata[idx]
                source = meta.get('source', 'Unknown Document')
                
                context_payload += f"--- Excerpt {idx_count} (Source: {source}) ---\n"
                context_payload += f"{content.strip()}\n\n"
                
            return context_payload
            
        except Exception as e:
            print(f"!!! [ERROR] RAG Retrieval Failed: {e}")
            return "Error retrieving market context."

if __name__ == "__main__":
    # Standalone test
    print("Testing Knowledge Engine Retrieval...")
    test_context = KnowledgeEngine.retrieve_context("What are the current investment trends in New Cairo?")
    print(test_context)

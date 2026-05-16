import os
import sys
from dotenv import load_dotenv

# Ensure cross-module imports work
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))

try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
except ImportError:
    pass  # Handled gracefully below

class KnowledgeEngine:
    """
    RAG Retrieval System for the Analyzer.
    Loads the local FAISS index built by the Data Acquisition layer 
    and performs semantic similarity searches to retrieve qualitative market context.
    """
    
    _vector_store = None
    _embeddings_model = None

    @classmethod
    def _initialize(cls):
        if cls._vector_store is not None:
            return True
            
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("!!! [WARNING] HUGGINGFACEHUB_API_TOKEN not found. RAG Knowledge Engine is offline.")
            return False
            
        try:
            # Re-initialize the same embedding model used during ingestion
            cls._embeddings_model = HuggingFaceEndpointEmbeddings(
                model="BAAI/bge-m3",
                huggingfacehub_api_token=hf_token
            )
            
            faiss_index_path = os.path.abspath(os.path.join(BASE_DIR, "4.Data", "3_market_knowledge", "faiss_index"))
            
            if not os.path.exists(faiss_index_path):
                print(f"!!! [WARNING] FAISS index not found at {faiss_index_path}. Run knowledge_loader.py first.")
                return False
                
            # Load the vector store into memory
            cls._vector_store = FAISS.load_local(faiss_index_path, cls._embeddings_model, allow_dangerous_deserialization=True)
            return True
        except Exception as e:
            print(f"!!! [ERROR] Failed to load RAG Vector Store: {e}")
            return False

    @classmethod
    def retrieve_context(cls, query: str, top_k: int = 3) -> str:
        """
        Executes a dense vector search against the market knowledge base.
        Returns a formatted string of the most relevant qualitative insights.
        """
        if not cls._initialize():
            return "No qualitative market context available at this time."
            
        try:
            # Perform similarity search
            relevant_docs = cls._vector_store.similarity_search(query, k=top_k)
            
            if not relevant_docs:
                return "No highly relevant market context found in the knowledge base."
                
            # Compile the retrieved chunks into a dense context block
            context_payload = "[RAG_MARKET_CONTEXT]\n"
            for idx, doc in enumerate(relevant_docs, start=1):
                source = doc.metadata.get('source', 'Unknown Document')
                # Optional: clean up source path to just the filename
                source_filename = os.path.basename(source) if source != 'Unknown Document' else source
                context_payload += f"--- Excerpt {idx} (Source: {source_filename}) ---\n"
                context_payload += f"{doc.page_content.strip()}\n\n"
                
            return context_payload
            
        except Exception as e:
            print(f"!!! [ERROR] RAG Retrieval Failed: {e}")
            return "Error retrieving market context."

if __name__ == "__main__":
    # Standalone test
    print("Testing Knowledge Engine Retrieval...")
    test_context = KnowledgeEngine.retrieve_context("What are the current investment trends in New Cairo?")
    print(test_context)

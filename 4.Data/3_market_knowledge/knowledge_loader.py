import os
import sys
from dotenv import load_dotenv

# Add paths to enable imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

# Load environment variables
load_dotenv(os.path.join(BASE_DIR, ".env"))

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    print("!!! [WARNING] HUGGINGFACEHUB_API_TOKEN not found in .env. Embeddings will fail.")

try:
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEndpointEmbeddings
except ImportError as e:
    print(f"!!! [ERROR] Missing dependency: {e}. Please run: pip install langchain-community langchain-huggingface faiss-cpu pypdf")
    sys.exit(1)

# Define paths
KNOWLEDGE_DIR = os.path.dirname(__file__)
RAW_DOCS_DIR = os.path.join(KNOWLEDGE_DIR, "raw_documents")
FAISS_INDEX_DIR = os.path.join(KNOWLEDGE_DIR, "faiss_index")

# Ensure raw docs directory exists
os.makedirs(RAW_DOCS_DIR, exist_ok=True)

def ingest_knowledge():
    print(f"--- [RAG Ingestion] Scanning {RAW_DOCS_DIR} for market reports ---")
    
    # Load PDFs and Text files
    pdf_loader = DirectoryLoader(RAW_DOCS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(RAW_DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    
    docs = []
    try:
        docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"[RAG Ingestion] PDF Loader Error: {e}")
        
    try:
        docs.extend(txt_loader.load())
    except Exception as e:
        print(f"[RAG Ingestion] Text Loader Error: {e}")
    
    if not docs:
        print("--- [RAG Ingestion] No documents found to index. Drop PDFs/txt files in raw_documents/ and run again. ---")
        # Still create an empty FAISS index so the engine doesn't crash on load
        empty_chunks = [
            # Dummy document to initialize the FAISS schema
            type('Document', (object,), {"page_content": "INIT_SYSTEM_DOC", "metadata": {}})()
        ]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        dummy_chunks = text_splitter.split_documents(empty_chunks)
        
        embeddings = HuggingFaceEndpointEmbeddings(model="BAAI/bge-m3", huggingfacehub_api_token=hf_token)
        vector_store = FAISS.from_documents(dummy_chunks, embeddings)
        vector_store.save_local(FAISS_INDEX_DIR)
        print(f"--- [RAG Ingestion] Created empty FAISS index at {FAISS_INDEX_DIR} ---")
        return
        
    print(f"--- [RAG Ingestion] Found {len(docs)} documents. Splitting text... ---")
    
    # Split documents into chunks for dense retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(docs)
    
    print(f"--- [RAG Ingestion] Generated {len(chunks)} chunks. Generating Embeddings using BAAI/bge-m3... ---")
    
    # Initialize the multilingual embedder
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=hf_token
    )
    
    # Build the FAISS Vector Index
    print("--- [RAG Ingestion] Building FAISS Vector Store... ---")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save the index locally
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"--- [RAG Ingestion] Successfully saved FAISS index to {FAISS_INDEX_DIR} ---")

if __name__ == "__main__":
    if not hf_token:
        print("Cannot run ingestion without HuggingFace API key.")
        sys.exit(1)
    ingest_knowledge()

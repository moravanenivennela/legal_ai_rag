import os
import re
import pickle
import pymupdf as fitz
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from rank_bm25 import BM25Okapi

# Configuration Parameters
DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
BM25_PKL_PATH = "./chroma_db/bm25_index.pkl"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # Alternative: "all-MiniLM-L6-v2"
COLLECTION_NAME = "legal_normative_docs"

def extract_pdf_with_metadata(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text from PDF pages and attempt to extract structural metadata."""
    doc = fitz.open(pdf_path)
    file_name = os.path.basename(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        
        # Regex heuristics to capture Article/Chapter references
        article_match = re.search(r'(Article\s+\d+|Art\.\s*\d+|Section\s+\d+|Chapter\s+[IVXLCDM\d]+)', text, re.IGNORECASE)
        citation = article_match.group(0) if article_match else f"Page {page_num + 1}"

        pages_data.append({
            "text": text,
            "metadata": {
                "source": file_name,
                "page": page_num + 1,
                "citation": citation
            }
        })
    return pages_data

def run_ingestion_pipeline():
    print("[1/4] Scanning local PDF legal files...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created {DATA_DIR}. Please drop your PDF legal documents into this folder and rerun.")
        return

    pdf_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}. Add legal PDFs to continue.")
        return

    raw_docs = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        raw_docs.extend(extract_pdf_with_metadata(pdf_path))

    print(f"[2/4] Chunking documents with RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", "Art.", "Article", ". ", " "]
    )

    documents = []
    metadatas = []
    ids = []
    corpus_for_bm25 = []

    chunk_id = 0
    for doc in raw_docs:
        chunks = text_splitter.split_text(doc["text"])
        for chunk in chunks:
            chunk_id += 1
            documents.append(chunk)
            metadatas.append(doc["metadata"])
            ids.append(f"doc_{chunk_id}")
            # Tokenization for BM25
            corpus_for_bm25.append(re.findall(r'\w+', chunk.lower()))

    print(f"[3/4] Initializing HuggingFace Embeddings ({EMBEDDING_MODEL_NAME})...")
    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("[4/4] Writing to ChromaDB and building BM25 index...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Reset existing collection if present
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2"}
    )

    # Compute dense embeddings and index in batches
    batch_size = 64
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        embeddings = embedding_fn.embed_documents(documents[i:end_idx])
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            embeddings=embeddings,
            metadatas=metadatas[i:end_idx]
        )

    # Build and serialize BM25 sparse index
    bm25 = BM25Okapi(corpus_for_bm25)
    with open(BM25_PKL_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "docs": documents, "metadatas": metadatas}, f)

    print(f" Successfully indexed {len(documents)} legal chunks locally!")

if __name__ == "__main__":
    run_ingestion_pipeline()
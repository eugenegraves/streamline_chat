import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
import sys

def load_faiss_index(index_path="faiss_index.bin", metadata_path="faiss_index_metadata.pkl"):
    """Load the FAISS index and metadata"""
    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors of dimension {index.d}")
    return index, metadata

def load_document_texts():
    """Load raw text from files in the knowledge base"""
    texts = []
    sources = []
    
    for filename in os.listdir("knowledge_base"):
        if filename.endswith(".txt"):
            file_path = os.path.join("knowledge_base", filename)
            with open(file_path, "r") as f:
                content = f.read().strip()
                texts.append(content)
                sources.append(file_path)
    
    return texts, sources

def find_most_similar_document(query, texts):
    """
    Find the document most similar to the query using simple text matching
    """
    # Simple approach: count word overlap
    query_words = set(query.lower().split())
    max_overlap = 0
    best_match = 0
    
    for i, text in enumerate(texts):
        text_words = set(text.lower().split())
        overlap = len(query_words.intersection(text_words))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = i
    
    return best_match

def search_by_id(index, document_id, top_k=3):
    """
    Use a document ID to search for similar documents
    """
    # Load existing embeddings
    embeddings = np.load("embeddings.npy")
    
    # Get the embedding for the document ID
    if document_id < 0 or document_id >= len(embeddings):
        raise ValueError(f"Document ID {document_id} is out of range")
    
    query_embedding = embeddings[document_id]
    
    # Make sure the embedding is normalized
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    
    # Search the index
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    return distances, indices

def process_query(query, top_k=3):
    """
    Process a user query by finding the most similar document and using its embedding
    
    Args:
        query: User query string
        top_k: Number of results to return
    """
    print(f"\n{'='*80}")
    print(f"USER QUERY: {query}")
    print(f"{'='*80}")
    
    # Load index and texts
    index, metadata = load_faiss_index()
    texts, sources = load_document_texts()
    
    # Find most similar document
    doc_id = find_most_similar_document(query, texts)
    print(f"Using document {doc_id} as proxy for the query")
    print(f"Document: {os.path.basename(sources[doc_id])}")
    
    # Search using the document embedding
    distances, indices = search_by_id(index, doc_id, top_k)
    
    # Print results
    print(f"\nSEARCH RESULTS:")
    print(f"{'='*80}")
    
    results = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx >= 0 and idx < len(texts):
            print(f"\nRank: {i+1}")
            print(f"Similarity: {distance:.4f}")
            print(f"Document: {os.path.basename(sources[idx])}")
            print(f"Text: {texts[idx]}")
            
            result = {
                "rank": i + 1,
                "document_id": int(idx),
                "similarity": float(distance),
                "document": sources[idx],
                "text": texts[idx]
            }
            results.append(result)
    
    return results

if __name__ == "__main__":
    # Check if a query was provided as a command-line argument
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        # Use a default query
        user_query = "How does clustering work in machine learning?"
    
    process_query(user_query) 
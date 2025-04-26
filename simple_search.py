import numpy as np
import faiss
import os
import pickle

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

def search_by_id(index, document_id, top_k=3):
    """
    Use a document ID to search for similar documents
    
    This avoids having to generate new embeddings for a query
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
    print(f"Searching for documents similar to document ID {document_id}...")
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    return distances, indices

def describe_document(document_id, texts, sources):
    """Print information about a document"""
    if document_id < 0 or document_id >= len(texts):
        print(f"Document ID {document_id} is out of range")
        return
    
    print(f"\nDocument ID: {document_id}")
    print(f"Source: {os.path.basename(sources[document_id])}")
    print(f"Content: {texts[document_id][:100]}...")

def main():
    """Main function to demonstrate the search functionality"""
    # Load the FAISS index and metadata
    index, metadata = load_faiss_index()
    
    # Load document texts
    texts, sources = load_document_texts()
    
    # Print information about all documents
    print("\nAvailable documents:")
    print("=" * 80)
    for i in range(len(texts)):
        describe_document(i, texts, sources)
    
    # Choose a document ID to use as a query
    query_id = 0  # Use the supervised learning document as a query
    
    print("\n" + "=" * 80)
    print(f"QUERY: Using document {query_id} as a query")
    print("=" * 80)
    
    # Search for similar documents
    distances, indices = search_by_id(index, query_id, top_k=3)
    
    # Print results
    print("\nSEARCH RESULTS:")
    print("=" * 80)
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx >= 0 and idx < len(texts):
            print(f"\nRank: {i+1}")
            print(f"Document ID: {idx}")
            print(f"Similarity: {distance:.4f}")
            print(f"Source: {os.path.basename(sources[idx])}")
            print(f"Content: {texts[idx][:200]}...")

if __name__ == "__main__":
    main() 
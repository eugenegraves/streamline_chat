import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import List, Dict, Any, Tuple
from langchain.schema import Document

def load_faiss_index(index_path: str = "faiss_index.bin", metadata_path: str = "faiss_index_metadata.pkl") -> Tuple[Any, Dict]:
    """
    Load a FAISS index and its metadata from files
    
    Args:
        index_path: Path to the FAISS index file
        metadata_path: Path to the metadata file
        
    Returns:
        Tuple of (FAISS index, metadata dictionary)
    """
    print(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(index_path)
    
    print(f"Loading metadata from {metadata_path}")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    print(f"Loaded index with {index.ntotal} vectors of dimension {index.d}")
    return index, metadata

def load_documents(directory: str = "knowledge_base") -> List[Document]:
    """
    Load all documents from the knowledge base
    
    Args:
        directory: Path to the directory containing the documents
        
    Returns:
        List of Document objects
    """
    # Initialize the text splitter (must match the one used to create embeddings)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    documents = []
    # Iterate through all text files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path)
            file_documents = loader.load()
            chunks = text_splitter.split_documents(file_documents)
            documents.extend(chunks)
    
    return documents

def semantic_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Perform semantic search on a user query using FAISS index
    
    Args:
        query: User query string
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing search results with text and metadata
    """
    # Load the FAISS index and metadata
    index, metadata = load_faiss_index()
    
    # Load documents to get the original text
    documents = load_documents()
    
    # Check if the number of documents matches the number of vectors in the index
    if len(documents) != index.ntotal:
        print(f"Warning: Number of documents ({len(documents)}) doesn't match number of vectors in the index ({index.ntotal})")
    
    # Initialize the sentence transformer model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    
    # Generate embedding for the query
    print(f"Generating embedding for query: '{query}'")
    query_embedding = model.encode([query])[0]
    
    # Normalize the query embedding for cosine similarity
    query_embedding_normalized = query_embedding.copy()
    faiss.normalize_L2(query_embedding_normalized.reshape(1, -1))
    
    # Search the index
    print(f"Searching for top {top_k} results...")
    distances, indices = index.search(query_embedding_normalized.reshape(1, -1), top_k)
    
    # Prepare results
    results = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx >= 0 and idx < len(documents):  # Ensure index is valid
            result = {
                "rank": i + 1,
                "text": documents[idx].page_content,
                "document": documents[idx].metadata["source"],
                "similarity": float(distance)
            }
            results.append(result)
    
    return results

def search_and_display(query: str, top_k: int = 3) -> None:
    """
    Perform semantic search and display results in a formatted way
    
    Args:
        query: User query string
        top_k: Number of top results to return
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")
    
    results = semantic_search(query, top_k)
    
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS:")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\nRank: {result['rank']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Document: {os.path.basename(result['document'])}")
        print(f"Text: {result['text'][:200]}...")
    
    return results

if __name__ == "__main__":
    # Example usage
    queries = [
        "What is supervised learning?",
        "How do neural networks work?",
        "Tell me about clustering algorithms",
        "What is feature engineering?"
    ]
    
    for query in queries:
        search_and_display(query) 
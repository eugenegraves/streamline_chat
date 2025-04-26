import numpy as np
import faiss
import os
import pickle

def create_faiss_index(embeddings_file="embeddings.npy", index_file="faiss_index.bin", metadata_file="faiss_index_metadata.pkl"):
    """
    Create a FAISS index from embeddings in a NumPy array and save it to a file
    
    Args:
        embeddings_file: Path to the NumPy file containing embeddings
        index_file: Path to save the FAISS index
        metadata_file: Path to save metadata (for linking indexed vectors back to documents)
    """
    print(f"Loading embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    # Print embedding information
    num_vectors, dimension = embeddings.shape
    print(f"Loaded {num_vectors} embeddings with dimension {dimension}")
    
    # Normalize vectors (optional but recommended for cosine similarity)
    faiss.normalize_L2(embeddings)
    print("Normalized embeddings for cosine similarity")
    
    # Create a FAISS index
    # We use IndexFlatIP which is exact search with inner product (cosine similarity)
    # For larger datasets, consider using approximate neighbors indices like IndexIVFFlat
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to the index
    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to the FAISS index")
    
    # Save the FAISS index to a file
    print(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, index_file)
    
    # Create metadata to map from index positions back to documents
    # In a real application, you would store document IDs or other metadata
    metadata = {
        "num_vectors": num_vectors,
        "dimension": dimension,
        "vector_ids": list(range(num_vectors))
    }
    
    # Save metadata
    print(f"Saving metadata to {metadata_file}")
    with open(metadata_file, "wb") as f:
        pickle.dump(metadata, f)
    
    print("FAISS index creation completed successfully!")
    
    return index, metadata

def test_search(index, query_embedding, top_k=3):
    """
    Test the FAISS index with a sample query
    
    Args:
        index: FAISS index
        query_embedding: Vector to search for
        top_k: Number of results to return
    """
    # Normalize the query vector
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    
    # Search the index
    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
    
    print(f"\nTesting search with a sample query:")
    print(f"Top {top_k} results:")
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        print(f"  {i+1}. Index: {idx}, Similarity: {distance:.4f}")
        
    return distances, indices

if __name__ == "__main__":
    # Check if embeddings file exists
    embeddings_file = "embeddings.npy"
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file {embeddings_file} not found. Please generate embeddings first.")
    
    # Create the FAISS index
    index, metadata = create_faiss_index(embeddings_file)
    
    # Test the index with a sample query (using the first embedding as a test)
    embeddings = np.load(embeddings_file)
    test_search(index, embeddings[0]) 
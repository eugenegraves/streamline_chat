import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Define the knowledge base directory
kb_dir = "knowledge_base"

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

# Function to load and chunk a single file
def process_file(file_path: str) -> List[Document]:
    loader = TextLoader(file_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    return chunks

# Function to get all text chunks from the knowledge base
def get_all_chunks() -> List[Document]:
    all_chunks = []
    
    # Iterate through all text files in the knowledge base
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_dir, filename)
            chunks = process_file(file_path)
            all_chunks.extend(chunks)
    
    return all_chunks

# Function to generate embeddings for text chunks
def generate_embeddings(chunks: List[Document]) -> np.ndarray:
    # Initialize the sentence transformer model
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    
    # Extract text content from chunks
    texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} chunks using {model_name}...")
    embeddings = model.encode(texts)
    
    return embeddings

def main():
    # Get all text chunks
    chunks = get_all_chunks()
    print(f"Loaded {len(chunks)} text chunks from the knowledge base")
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    
    # Save embeddings as NumPy array
    save_path = "embeddings.npy"
    np.save(save_path, embeddings)
    print(f"Embeddings saved to {save_path}")
    
    # Print information about the embeddings
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Example embedding (first 5 dimensions): {embeddings[0][:5]}")
    
    # Associate chunks with their embeddings
    chunk_embedding_pairs = list(zip(chunks, embeddings))
    
    return chunks, embeddings, chunk_embedding_pairs

if __name__ == "__main__":
    chunks, embeddings, chunk_embedding_pairs = main() 
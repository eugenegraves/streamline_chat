import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from typing import List
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

# Main function to process all files in the knowledge base
def main():
    all_chunks = []
    file_count = 0
    
    # Iterate through all text files in the knowledge base
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(kb_dir, filename)
            print(f"Processing {filename}...")
            
            # Load and chunk the file
            chunks = process_file(file_path)
            all_chunks.extend(chunks)
            file_count += 1
            
            # Print information about the chunks
            print(f"  - Created {len(chunks)} chunks")
    
    # Summary of processing
    print(f"\nProcessed {file_count} files")
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Example of accessing chunks
    if all_chunks:
        print("\nExample of a chunk:")
        print(f"Content: {all_chunks[0].page_content[:100]}...")
        print(f"Source: {all_chunks[0].metadata['source']}")
    
    return all_chunks

if __name__ == "__main__":
    chunks = main() 
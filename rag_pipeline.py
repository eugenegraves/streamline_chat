import os
import torch
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, DirectoryLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import argparse

def create_langchain_pipeline(model_name="gpt2", max_new_tokens=100):
    """
    Create a LangChain pipeline from a Hugging Face model
    """
    print(f"Loading {model_name} model and tokenizer...")
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Check if CUDA is available and move model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully and moved to {device}")
    
    # Create a transformers pipeline for text generation
    # Use max_new_tokens instead of max_length to avoid the input length issue
    text_generation = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,  # This controls how many new tokens to generate
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Create a LangChain wrapper
    llm = HuggingFacePipeline(pipeline=text_generation)
    
    print(f"LangChain pipeline created successfully")
    return llm

def load_documents(data_dir="data"):
    """
    Load documents from a directory
    """
    print(f"Loading documents from {data_dir}...")
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def create_vector_store(documents, embeddings_model_name="all-MiniLM-L6-v2"):
    """
    Create a FAISS vector store from documents
    """
    print(f"Creating vector store using {embeddings_model_name} embeddings...")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks")
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Created vector store with {len(chunks)} chunks")
    return vector_store

def create_rag_pipeline(vector_store, llm):
    """
    Create a RAG pipeline using vector store for retrieval and LLM for generation
    """
    print("Creating RAG pipeline...")
    
    # Define a shorter prompt template to avoid exceeding token limits
    template = """Using the following context, please answer the question.
Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create RAG chain with an appropriate chunk size to limit context length
    try:
        # Try using the newer API
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),  # Limit to 2 docs to reduce context size
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    except TypeError:
        # Fallback to older API if needed
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    return rag_chain

def process_query(query, rag_chain):
    """
    Process a query using the RAG pipeline
    """
    print(f"Processing query: {query}")
    
    try:
        # Try newer API
        response = rag_chain.invoke({"query": query})
    except (AttributeError, TypeError) as e:
        print(f"Falling back to older API due to: {e}")
        # Fall back to older API
        response = rag_chain({"query": query})
    
    # Extract result based on the response structure
    if isinstance(response, str):
        answer = response
    elif isinstance(response, dict):
        if "result" in response:
            answer = response["result"]
        elif "answer" in response:
            answer = response["answer"]
        else:
            # Try to extract from another key if the structure is different
            answer = str(response.get("output", str(response)))
    else:
        answer = str(response)
    
    # Extract source documents if available
    if isinstance(response, dict):
        source_docs = response.get("source_documents", [])
    else:
        source_docs = []
    
    # Format sources
    sources = []
    for i, doc in enumerate(source_docs):
        if hasattr(doc, "metadata"):
            source = f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}"
            sources.append(source)
    
    formatted_response = {
        "answer": answer,
        "sources": sources
    }
    
    return formatted_response

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Pipeline with GPT-2")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing documents")
    parser.add_argument("--max_new_tokens", type=int, default=150, help="Maximum number of tokens to generate")
    args = parser.parse_args()
    
    # Setup the RAG pipeline
    documents = load_documents(args.data_dir)
    vector_store = create_vector_store(documents)
    llm = create_langchain_pipeline(max_new_tokens=args.max_new_tokens)
    rag_chain = create_rag_pipeline(vector_store, llm)
    
    # Process the query if provided
    if args.query:
        result = process_query(args.query, rag_chain)
        print("\n" + "="*80)
        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        for source in result["sources"]:
            print(source)
        print("="*80)
    else:
        # Interactive mode
        print("\nEnter 'exit' to quit")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
                
            result = process_query(query, rag_chain)
            print("\n" + "="*80)
            print("ANSWER:")
            print(result["answer"])
            print("\nSOURCES:")
            for source in result["sources"]:
                print(source)
            print("="*80)

if __name__ == "__main__":
    # Set transformers cache location
    os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"
    
    main() 
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import components from our RAG pipeline
from rag_pipeline import (
    load_documents,
    create_vector_store,
    create_langchain_pipeline,
    create_rag_pipeline,
    process_query
)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="API for a Retrieval-Augmented Generation (RAG) pipeline using GPT-2",
    version="1.0.0"
)

# Define request and response models
class QueryRequest(BaseModel):
    question: str
    max_new_tokens: Optional[int] = 150

class Source(BaseModel):
    name: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Global variables to store our RAG pipeline components
documents = None
vector_store = None
llm = None
rag_chain = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline components when the API starts"""
    global documents, vector_store, llm, rag_chain
    
    # Set transformers cache location
    os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"
    
    print("Loading RAG pipeline components...")
    documents = load_documents("data")
    vector_store = create_vector_store(documents)
    llm = create_langchain_pipeline(max_new_tokens=150)
    rag_chain = create_rag_pipeline(vector_store, llm)
    print("RAG pipeline initialized and ready!")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process a question using the RAG pipeline
    
    Parameters:
    - question: The question to answer
    - max_new_tokens: Maximum number of tokens to generate (optional)
    
    Returns:
    - answer: The generated answer
    - sources: List of source documents used
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized. Please try again later."
        )
    
    try:
        # If a different max_new_tokens is requested, recreate the LLM
        if request.max_new_tokens != 150:
            global llm
            print(f"Recreating LLM with max_new_tokens={request.max_new_tokens}")
            llm = create_langchain_pipeline(max_new_tokens=request.max_new_tokens)
            rag_chain = create_rag_pipeline(vector_store, llm)
        
        # Process the query
        result = process_query(request.question, rag_chain)
        
        # Return the result
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG Pipeline API is running",
        "usage": "Send a POST request to /query with a JSON payload containing a 'question' field"
    }

def main():
    """Run the FastAPI app with Uvicorn"""
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main() 
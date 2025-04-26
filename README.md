# RAG Pipeline with GPT-2

This repository contains a Retrieval-Augmented Generation (RAG) pipeline that uses FAISS vector store for document retrieval and the GPT-2 model for text generation.

## Features

- Uses LangChain for creating the RAG pipeline
- Retrieves relevant documents using FAISS vector store
- Generates answers using Hugging Face's GPT-2 model
- Provides source attribution for answers
- Supports interactive query mode
- Exposes the RAG pipeline through a FastAPI REST API
- Includes a user-friendly Streamlit web interface

## Requirements

Required packages are listed in `requirements.txt`:

```
langchain==0.0.267
sentence-transformers==2.2.2
faiss-cpu==1.7.4
transformers>=4.15.0
fastapi==0.103.1
uvicorn==0.23.2
streamlit==1.26.0
python-dotenv==1.0.0
torch>=1.10.0
huggingface-hub==0.19.4
requests>=2.28.0
```

Install the requirements with:

```bash
pip install -r requirements.txt
```

## Command Line Usage

### Command Line Interface

```bash
python rag_pipeline.py --query "Your question here" --max_new_tokens 150
```

Arguments:
- `--query`: The question you want to ask
- `--data_dir`: Directory containing documents (default: "data")
- `--max_new_tokens`: Maximum number of tokens to generate (default: 150)

### Interactive Mode

If you don't provide a query, the script will run in interactive mode:

```bash
python rag_pipeline.py
```

Then you can enter questions interactively.

## API Usage

### Starting the API Server

To start the FastAPI server:

```bash
python api.py
```

This will start the server on http://0.0.0.0:8000. You can access the API documentation at http://localhost:8000/docs.

### API Endpoints

- `GET /`: Root endpoint with API information
- `POST /query`: Process a question using the RAG pipeline

Example request to `/query`:

```json
{
  "question": "What are neural networks?",
  "max_new_tokens": 150
}
```

Example response:

```json
{
  "answer": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains...",
  "sources": [
    "Source 1: data/neural_networks.txt"
  ]
}
```

### Using the Client Script

A client script is provided to interact with the API:

```bash
python client.py --question "What are neural networks?" --max-new-tokens 150
```

Arguments:
- `--question`: The question to ask (required)
- `--max-new-tokens`: Maximum number of tokens to generate (default: 150)
- `--url`: The base URL of the API (default: http://localhost:8000)

## Streamlit Web Interface

A user-friendly web interface is available through Streamlit:

### Starting the Streamlit App

```bash
streamlit run streamlit_app.py
```

This will start the Streamlit app on http://localhost:8501 by default.

### Features of the Streamlit App

- Simple text input for questions
- Configuration options in the sidebar
- Displays answers with source attribution
- Keeps a history of previous questions and answers
- Shows loading indicator during processing
- Error handling for API connection issues

### Using the Streamlit App

1. Start the FastAPI server: `python api.py`
2. In a separate terminal, start the Streamlit app: `streamlit run streamlit_app.py`
3. Open your browser to http://localhost:8501
4. Enter your question and click "Submit Question"

## Components

1. **Document Loading**: Loads text documents from the specified directory.

2. **Vector Store Creation**: 
   - Splits documents into chunks 
   - Creates embeddings using HuggingFaceEmbeddings
   - Stores document chunks in a FAISS vector store

3. **Language Model**: 
   - Uses GPT-2 for text generation
   - Configurable with parameters like max_new_tokens

4. **RAG Pipeline**:
   - Retrieves relevant document chunks for a query
   - Passes the context and query to the language model
   - Returns a generated answer with source information

5. **FastAPI Application**:
   - Exposes the RAG pipeline as a REST API
   - Provides API documentation with Swagger UI
   - Handles loading the pipeline at startup

6. **Streamlit Web Interface**:
   - Provides a user-friendly interface for asking questions
   - Communicates with the FastAPI backend
   - Displays answers and sources in a readable format
   - Maintains a session history of questions and answers

## Examples

Example query about neural networks:

```bash
python rag_pipeline.py --query "What are neural networks and how do they work?" --max_new_tokens 150
```

Example API query using curl:

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What is Retrieval-Augmented Generation (RAG)?",
  "max_new_tokens": 200
}'
```

## Adding Your Own Documents

Place your text documents in the `data` directory to include them in the knowledge base. The documents will be automatically loaded, chunked, and indexed when the script runs.

# RAG Pipeline with GPT-2

This repository contains a Retrieval-Augmented Generation (RAG) pipeline that uses FAISS vector store for document retrieval and the GPT-2 model for text generation.

## Features

- Uses LangChain for creating the RAG pipeline
- Retrieves relevant documents using FAISS vector store
- Generates answers using Hugging Face's GPT-2 model
- Provides source attribution for answers
- Supports interactive query mode

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
```

Install the requirements with:

```bash
pip install -r requirements.txt
```

## Usage

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

## Examples

Example query about neural networks:

```bash
python rag_pipeline.py --query "What are neural networks and how do they work?" --max_new_tokens 150
```

Example query about RAG:

```bash
python rag_pipeline.py --query "What is Retrieval-Augmented Generation (RAG)?" --max_new_tokens 200
```

## Adding Your Own Documents

Place your text documents in the `data` directory to include them in the knowledge base. The documents will be automatically loaded, chunked, and indexed when the script runs.

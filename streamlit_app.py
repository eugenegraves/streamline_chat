import streamlit as st
import requests
import json

# Set page title and description
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🔍",
    layout="wide"
)

def query_rag_api(question, max_new_tokens=150, api_url="http://localhost:8000/query"):
    """
    Send a query to the RAG API endpoint
    
    Args:
        question: The question to ask
        max_new_tokens: Maximum number of tokens to generate
        api_url: URL of the API endpoint
        
    Returns:
        Response from the API or error message
    """
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "question": question,
            "max_new_tokens": max_new_tokens
        }
        
        # Send POST request to the API
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"Error: {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "error": "Connection Error",
            "details": str(e)
        }

# Create the Streamlit UI
st.title("🔍 RAG Q&A System")
st.markdown("""
This app uses a Retrieval-Augmented Generation (RAG) pipeline to answer questions
based on a knowledge base of documents. The app connects to a FastAPI backend
that retrieves relevant information and generates answers using GPT-2.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000/query",
    help="URL of the RAG API endpoint"
)

max_tokens = st.sidebar.slider(
    "Max New Tokens",
    min_value=50,
    max_value=500,
    value=150,
    step=10,
    help="Maximum number of tokens to generate in the answer"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This app connects to a RAG pipeline with:
- FAISS vector store for retrieval
- GPT-2 for text generation
- FastAPI backend

Source documents include information on:
- Machine Learning
- Neural Networks
- RAG Systems
""")

# Create main input area
st.markdown("### Ask a Question")
user_question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What are neural networks and how do they work?"
)

# Add a submit button
submit_button = st.button("Submit Question")

# Process the question when submitted
if submit_button and user_question:
    with st.spinner("Generating answer..."):
        # Display a loading message while waiting for the API
        response = query_rag_api(user_question, max_tokens, api_url)
        
        if "error" in response:
            st.error(f"Error: {response['error']}")
            st.error(f"Details: {response.get('details', 'No details provided')}")
            st.warning("Make sure the FastAPI server is running at the specified URL.")
        else:
            # Display the answer
            st.markdown("### Answer")
            st.write(response["answer"])
            
            # Display the sources
            st.markdown("### Sources")
            for source in response["sources"]:
                st.markdown(f"- {source}")
                
            # Add a visual separator
            st.markdown("---")
            
            # Save to history
            if "question_history" not in st.session_state:
                st.session_state.question_history = []
                
            # Add current Q&A to history (limit to 5 most recent)
            st.session_state.question_history.insert(0, {
                "question": user_question,
                "answer": response["answer"],
                "sources": response["sources"]
            })
            
            if len(st.session_state.question_history) > 5:
                st.session_state.question_history = st.session_state.question_history[:5]

# Display history if it exists
if "question_history" in st.session_state and len(st.session_state.question_history) > 0:
    st.markdown("### Previous Questions")
    
    for i, qa in enumerate(st.session_state.question_history):
        with st.expander(f"Q: {qa['question']}"):
            st.markdown("**Answer:**")
            st.write(qa["answer"])
            
            st.markdown("**Sources:**")
            for source in qa["sources"]:
                st.markdown(f"- {source}")

# Add footer
st.markdown("---")
st.markdown("📚 RAG Q&A System - Powered by LangChain, FAISS, and GPT-2") 
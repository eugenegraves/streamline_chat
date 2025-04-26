import requests
import json
import argparse

def query_rag_api(question, max_new_tokens=150, url="http://localhost:8000"):
    """
    Query the RAG API with a question
    
    Args:
        question: The question to ask
        max_new_tokens: Maximum number of tokens to generate (default: 150)
        url: The base URL of the API (default: http://localhost:8000)
        
    Returns:
        The API response
    """
    # Prepare the request
    endpoint = f"{url}/query"
    payload = {
        "question": question,
        "max_new_tokens": max_new_tokens
    }
    headers = {"Content-Type": "application/json"}
    
    # Make the request
    print(f"Sending query to {endpoint}...")
    response = requests.post(endpoint, data=json.dumps(payload), headers=headers)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"Error: {response.status_code} - {response.text}"
        print(error_msg)
        return {"error": error_msg}

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Client for the RAG API")
    parser.add_argument("--question", type=str, required=True, help="The question to ask")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Maximum number of tokens to generate")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="The base URL of the API")
    args = parser.parse_args()
    
    # Query the API
    result = query_rag_api(args.question, args.max_new_tokens, args.url)
    
    # Print the result
    if "error" not in result:
        print("\n" + "="*80)
        print("ANSWER:")
        print(result["answer"])
        print("\nSOURCES:")
        for source in result["sources"]:
            print(source)
        print("="*80)
    else:
        print(f"Failed to get response: {result['error']}")

if __name__ == "__main__":
    main() 
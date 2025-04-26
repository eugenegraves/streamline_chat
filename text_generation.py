import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

def load_gpt2_model(model_name="gpt2", max_length=100):
    """
    Load a GPT-2 model for text generation using Hugging Face transformers
    
    Args:
        model_name: Name or path of the model to load
        max_length: Maximum length of generated text
        
    Returns:
        Transformers pipeline for text generation
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
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )
    
    print(f"Text generation pipeline created successfully")
    return pipe

def generate_text(generation_pipeline, prompt, max_length=100, num_return_sequences=1):
    """
    Generate text based on a prompt
    
    Args:
        generation_pipeline: Hugging Face pipeline for text generation
        prompt: Input text to generate from
        max_length: Maximum length of text to generate
        num_return_sequences: Number of different sequences to generate
        
    Returns:
        Generated text
    """
    print(f"Generating text for prompt: '{prompt}'")
    
    results = generation_pipeline(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences
    )
    
    # Extract the generated text from the results
    if num_return_sequences == 1:
        return results[0]['generated_text']
    else:
        return [result['generated_text'] for result in results]

def run_examples():
    """Run example text generation to demonstrate the model"""
    # Load the model
    generator = load_gpt2_model()
    
    # Example prompts for text generation
    examples = [
        "Machine learning is a field that",
        "Neural networks function by",
        "Clustering in data analysis involves",
        "The benefits of supervised learning include",
    ]
    
    # Generate text for each example
    for i, prompt in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"Example {i+1}: {prompt}")
        print(f"{'='*80}")
        
        result = generate_text(generator, prompt)
        
        print(f"\nGenerated text:")
        print(f"{'='*40}")
        print(result)
        print(f"{'='*40}")

if __name__ == "__main__":
    # Set a smaller model size to reduce memory usage if needed
    os.environ["TRANSFORMERS_CACHE"] = "./transformers_cache"
    
    run_examples() 
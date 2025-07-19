#!/usr/bin/env python3
"""
Check model type (embedding or reranking) for NoEmbed service
"""

import sys
import requests
import json

def check_model_type(base_url, model_name):
    """Check if a model is for embeddings or reranking"""
    try:
        # Get models list
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        
        models = response.json()["data"]
        
        # Find the requested model
        for model in models:
            if model["id"] == model_name:
                model_type = model.get("type", "unknown")
                capabilities = model.get("capabilities", {})
                
                print(f"\nModel: {model_name}")
                print(f"Type: {model_type}")
                print(f"Capabilities:")
                print(f"  - Embeddings: {'✓' if capabilities.get('embeddings') else '✗'}")
                print(f"  - Reranking: {'✓' if capabilities.get('reranking') else '✗'}")
                
                if model_type == "embedding":
                    print(f"\nUse this endpoint: POST {base_url}/v1/embeddings")
                    print("Example:")
                    print(f'curl -X POST {base_url}/v1/embeddings \\')
                    print('  -H "Content-Type: application/json" \\')
                    print(f'  -d \'{{"model": "{model_name}", "input": "Your text here"}}\'')
                elif model_type == "reranking":
                    print(f"\nUse this endpoint: POST {base_url}/v1/rerank")
                    print("Example:")
                    print(f'curl -X POST {base_url}/v1/rerank \\')
                    print('  -H "Content-Type: application/json" \\')
                    print(f'  -d \'{{"model": "{model_name}", "query": "Your query", "documents": ["doc1", "doc2"]}}\'')
                
                return
        
        print(f"\nModel '{model_name}' not found!")
        print("\nAvailable models:")
        embeddings = [m["id"] for m in models if m.get("type") == "embedding"]
        rerankers = [m["id"] for m in models if m.get("type") == "reranking"]
        
        if embeddings:
            print("\nEmbedding models:")
            for m in sorted(embeddings):
                print(f"  - {m}")
        
        if rerankers:
            print("\nReranking models:")
            for m in sorted(rerankers):
                print(f"  - {m}")
                
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_model_type.py <model_name> [base_url]")
        print("Example: python check_model_type.py mmarco-minilm-l12")
        print("Example: python check_model_type.py norbert2 http://localhost:7000")
        sys.exit(1)
    
    model_name = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:7000"
    
    check_model_type(base_url, model_name)
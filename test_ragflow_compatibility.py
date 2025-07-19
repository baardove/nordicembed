#!/usr/bin/env python3
"""
Test RAGFlow compatibility for NoEmbed models
"""

import sys
import requests
import json

def test_model_with_embeddings_endpoint(base_url, model_name):
    """Test if a model works with the /v1/embeddings endpoint (RAGFlow compatible)"""
    
    endpoint = f"{base_url}/v1/embeddings"
    test_text = "This is a test text for embeddings"
    
    try:
        response = requests.post(endpoint, json={
            "model": model_name,
            "input": test_text
        })
        
        if response.status_code == 200:
            data = response.json()
            embedding_length = len(data["data"][0]["embedding"])
            print(f"✅ {model_name}: Compatible with RAGFlow (embedding dim: {embedding_length})")
            return True
        else:
            error = response.json().get("detail", "Unknown error")
            if "reranking model" in error:
                print(f"❌ {model_name}: NOT compatible - This is a reranking model")
            else:
                print(f"❌ {model_name}: NOT compatible - {error}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name}: Error testing - {str(e)}")
        return False

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7000"
    
    print("Testing RAGFlow Compatibility for NoEmbed Models")
    print("=" * 50)
    print(f"Base URL: {base_url}")
    print()
    
    # Get all available models
    try:
        response = requests.get(f"{base_url}/v1/models")
        models = response.json()["data"]
    except:
        print("Error: Could not fetch models list")
        sys.exit(1)
    
    # Separate models by type
    embedding_models = []
    reranking_models = []
    
    for model in models:
        if model.get("type") == "embedding":
            embedding_models.append(model["id"])
        elif model.get("type") == "reranking":
            reranking_models.append(model["id"])
    
    # Test embedding models
    print("EMBEDDING MODELS (Should work with RAGFlow):")
    print("-" * 50)
    compatible_count = 0
    for model in sorted(set(embedding_models)):
        if test_model_with_embeddings_endpoint(base_url, model):
            compatible_count += 1
    
    print()
    print("RERANKING MODELS (Will NOT work with RAGFlow):")
    print("-" * 50)
    for model in sorted(set(reranking_models)):
        test_model_with_embeddings_endpoint(base_url, model)
    
    print()
    print("SUMMARY:")
    print("-" * 50)
    print(f"✅ {compatible_count} embedding models are compatible with RAGFlow")
    print(f"❌ {len(set(reranking_models))} reranking models are NOT compatible with RAGFlow")
    print()
    print("For RAGFlow, use ONLY the embedding models marked with ✅")

if __name__ == "__main__":
    main()
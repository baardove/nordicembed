#!/usr/bin/env python3
"""
Compare different Norwegian embedding models.
This script demonstrates how to switch between models and compare their outputs.
"""
import requests
import numpy as np
import time
import json
from typing import List, Dict

# Configuration
API_URL = "http://localhost:6000"

# Test sentences for comparison
TEST_SENTENCES = [
    "Norge er et vakkert land med fjorder og fjell.",
    "Oslo er hovedstaden i Norge og ligger ved Oslofjorden.",
    "Jeg liker å gå på ski om vinteren.",
    "Nordmenn er glade i friluftsliv og natur.",
    "Kaffe er en viktig del av norsk kultur."
]

# Models to compare
MODELS_TO_COMPARE = [
    {
        "name": "norbert2",
        "description": "General purpose Norwegian BERT"
    },
    {
        "name": "nb-bert-base",
        "description": "National Library of Norway BERT base"
    },
    {
        "name": "simcse-nb-bert-large",
        "description": "Optimized for semantic similarity"
    },
    {
        "name": "electra-small-nordic",
        "description": "Fast and lightweight Nordic model"
    }
]


def get_embeddings(texts: List[str], api_url: str = API_URL) -> List[List[float]]:
    """Get embeddings from the API."""
    response = requests.post(
        f"{api_url}/api/embed",
        json={"texts": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))


def compare_models():
    """Compare different models on the same texts."""
    print("Norwegian Embedding Models Comparison")
    print("=" * 80)
    print()
    
    # Check current model
    info_response = requests.get(f"{API_URL}/api/info")
    current_model = info_response.json()["model"]
    print(f"Current model: {current_model}")
    print()
    
    print("Test sentences:")
    for i, sentence in enumerate(TEST_SENTENCES):
        print(f"{i+1}. {sentence}")
    print()
    
    # For each model, calculate embeddings and similarities
    for model_info in MODELS_TO_COMPARE:
        model_name = model_info["name"]
        
        if model_name != current_model:
            print(f"\nNote: To test {model_name}, update MODEL_NAME in .env and restart the service")
            continue
        
        print(f"\nModel: {model_name}")
        print(f"Description: {model_info['description']}")
        print("-" * 40)
        
        # Measure embedding time
        start_time = time.time()
        embeddings = get_embeddings(TEST_SENTENCES)
        elapsed_time = time.time() - start_time
        
        print(f"Embedding time: {elapsed_time*1000:.2f} ms for {len(TEST_SENTENCES)} texts")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Calculate similarity matrix
        print("\nSimilarity matrix:")
        print("     ", end="")
        for i in range(len(TEST_SENTENCES)):
            print(f"  S{i+1}  ", end="")
        print()
        
        similarities = []
        for i in range(len(TEST_SENTENCES)):
            print(f"S{i+1}: ", end="")
            row_sims = []
            for j in range(len(TEST_SENTENCES)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                row_sims.append(sim)
                print(f"{sim:6.3f}", end="")
            similarities.append(row_sims)
            print()
        
        # Find most similar pairs (excluding self-similarity)
        print("\nMost similar sentence pairs:")
        pairs = []
        for i in range(len(TEST_SENTENCES)):
            for j in range(i+1, len(TEST_SENTENCES)):
                pairs.append((similarities[i][j], i, j))
        
        pairs.sort(reverse=True)
        for sim, i, j in pairs[:3]:
            print(f"  S{i+1} <-> S{j+1}: {sim:.3f}")


def demonstrate_use_cases():
    """Demonstrate specific use cases for different models."""
    print("\n" + "=" * 80)
    print("Model Use Case Recommendations")
    print("=" * 80)
    
    use_cases = [
        {
            "scenario": "Semantic Search Application",
            "recommended": ["simcse-nb-bert-large", "sentence-bert-base"],
            "reason": "These models are specifically trained for sentence similarity"
        },
        {
            "scenario": "Real-time API with Low Latency Requirements",
            "recommended": ["electra-small-nordic", "norbert3-base"],
            "reason": "Smaller models with faster inference times"
        },
        {
            "scenario": "High-Quality Document Analysis",
            "recommended": ["norbert3-large", "nb-bert-large"],
            "reason": "Larger models capture more nuanced semantic relationships"
        },
        {
            "scenario": "Multilingual Application (Norwegian + other languages)",
            "recommended": ["xlm-roberta-base"],
            "reason": "Trained on 100+ languages while maintaining good Norwegian performance"
        }
    ]
    
    for use_case in use_cases:
        print(f"\n{use_case['scenario']}:")
        print(f"  Recommended models: {', '.join(use_case['recommended'])}")
        print(f"  Reason: {use_case['reason']}")


if __name__ == "__main__":
    try:
        compare_models()
        demonstrate_use_cases()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the embedding service.")
        print("Make sure the service is running at", API_URL)
    except Exception as e:
        print(f"Error: {e}")
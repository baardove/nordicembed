#!/usr/bin/env python3
"""
Test script for the Norwegian Embedding Service API.
"""
import requests
import json
import time
import argparse
import sys

def test_health(base_url):
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_info(base_url):
    """Test info endpoint."""
    print("\nTesting info endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Service info: {data}")
        return True
    except Exception as e:
        print(f"❌ Info endpoint failed: {e}")
        return False

def test_single_embedding(base_url):
    """Test single text embedding."""
    print("\nTesting single text embedding...")
    text = "Dette er en test av den norske embedding-tjenesten."
    
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"texts": [text]}
        )
        response.raise_for_status()
        data = response.json()
        
        embeddings = data["embeddings"]
        print(f"✅ Single embedding generated")
        print(f"   Text: {text}")
        print(f"   Embedding size: {len(embeddings[0])}")
        print(f"   Sample (first 5 dims): {embeddings[0][:5]}")
        return True
    except Exception as e:
        print(f"❌ Single embedding failed: {e}")
        return False

def test_batch_embedding(base_url):
    """Test batch embedding."""
    print("\nTesting batch embedding...")
    texts = [
        "Hvordan har du det i dag?",
        "Jeg liker å lese norske bøker.",
        "Bergen er en vakker by på vestlandet.",
        "Klimaendringer påvirker hele verden."
    ]
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/embed",
            json={"texts": texts}
        )
        response.raise_for_status()
        elapsed = time.time() - start_time
        
        data = response.json()
        embeddings = data["embeddings"]
        
        print(f"✅ Batch embedding generated")
        print(f"   Batch size: {len(texts)}")
        print(f"   Time taken: {elapsed:.2f}s")
        print(f"   Embeddings generated: {len(embeddings)}")
        
        # Verify all embeddings have same dimension
        dims = [len(emb) for emb in embeddings]
        if len(set(dims)) == 1:
            print(f"   ✅ All embeddings have consistent dimension: {dims[0]}")
        else:
            print(f"   ❌ Inconsistent dimensions: {dims}")
            
        return True
    except Exception as e:
        print(f"❌ Batch embedding failed: {e}")
        return False

def test_similarity(base_url):
    """Test semantic similarity with embeddings."""
    print("\nTesting semantic similarity...")
    texts = [
        "Jeg liker å spise pizza.",
        "Pizza er min favorittmat.",
        "Fotball er en populær sport i Norge.",
        "Skiing er populært om vinteren."
    ]
    
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"texts": texts}
        )
        response.raise_for_status()
        embeddings = response.json()["embeddings"]
        
        # Calculate cosine similarity
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        print("✅ Similarity scores (cosine):")
        print(f"   'Pizza' texts (0 vs 1): {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
        print(f"   'Pizza' vs 'Football' (0 vs 2): {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
        print(f"   'Sports' texts (2 vs 3): {cosine_similarity(embeddings[2], embeddings[3]):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Similarity test failed: {e}")
        return False

def test_error_handling(base_url):
    """Test error handling."""
    print("\nTesting error handling...")
    
    # Test empty texts
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"texts": []}
        )
        if response.status_code == 400:
            print("✅ Empty texts correctly rejected")
        else:
            print("❌ Empty texts should return 400")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test invalid request
    try:
        response = requests.post(
            f"{base_url}/embed",
            json={"invalid": "data"}
        )
        if response.status_code in [400, 422]:
            print("✅ Invalid request correctly rejected")
        else:
            print("❌ Invalid request should return 400/422")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test Norwegian Embedding Service")
    parser.add_argument(
        "--url",
        default="http://localhost:6000",
        help="Base URL of the embedding service"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    print(f"Testing Norwegian Embedding Service at: {args.url}")
    print("=" * 50)
    
    tests = [
        test_health,
        test_info,
        test_single_embedding,
        test_batch_embedding,
        test_similarity,
        test_error_handling
    ]
    
    passed = 0
    for test in tests:
        if test(args.url):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed < len(tests):
        sys.exit(1)

if __name__ == "__main__":
    main()
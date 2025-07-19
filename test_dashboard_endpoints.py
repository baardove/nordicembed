#!/usr/bin/env python3
"""Test dashboard endpoints to ensure they're working"""

import requests
import json

BASE_URL = "http://localhost:7000"

def test_embedding_endpoint():
    """Test the embedding endpoint used by dashboard"""
    print("Testing Embedding Endpoint...")
    
    response = requests.post(f"{BASE_URL}/api/test-embed", json={
        "texts": ["Dette er en test"],
        "model": "norbert2",
        "pooling_strategy": "mean"
    })
    
    if response.status_code == 200:
        data = response.json()
        embedding_dims = len(data["embeddings"][0])
        print(f"✅ Embedding test successful! Dimensions: {embedding_dims}")
    else:
        print(f"❌ Embedding test failed: {response.status_code}")
        print(f"Error: {response.text}")

def test_reranking_endpoint():
    """Test the reranking endpoint used by dashboard"""
    print("\nTesting Reranking Endpoint...")
    
    response = requests.post(f"{BASE_URL}/api/rerank", json={
        "query": "Hva er hovedstaden i Norge?",
        "documents": [
            "Oslo er hovedstaden i Norge",
            "Bergen er en by i Norge"
        ],
        "model": "mmarco-minilm-l12",
        "top_k": 2
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Reranking test successful!")
        print(f"Results: {len(data['results'])} documents reranked")
        for result in data['results']:
            print(f"  - Score: {result['score']:.4f}, Doc: {result['document'][:50]}...")
    else:
        print(f"❌ Reranking test failed: {response.status_code}")
        print(f"Error: {response.text}")

def test_dashboard_loading():
    """Test if dashboard loads"""
    print("\nTesting Dashboard Loading...")
    
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200 and "Test Embeddings" in response.text:
        print("✅ Dashboard loads successfully")
    else:
        print("❌ Dashboard failed to load")

def test_javascript_loading():
    """Test if JavaScript loads"""
    print("\nTesting JavaScript Loading...")
    
    response = requests.get(f"{BASE_URL}/static/dashboard.js")
    if response.status_code == 200 and "testEmbeddings" in response.text:
        print("✅ JavaScript loads successfully")
    else:
        print("❌ JavaScript failed to load")

if __name__ == "__main__":
    print("Testing NoEmbed Dashboard Endpoints")
    print("=" * 40)
    
    test_dashboard_loading()
    test_javascript_loading()
    test_embedding_endpoint()
    test_reranking_endpoint()
    
    print("\n" + "=" * 40)
    print("If all tests pass, the issue might be:")
    print("1. Browser JavaScript errors (check browser console)")
    print("2. CORS issues (check browser network tab)")
    print("3. Form submission not prevented (check event.preventDefault())")
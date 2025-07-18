#!/usr/bin/env python3
"""
Example of using the Norwegian Embedding Service in Python.
"""
import requests
import numpy as np

# Configuration
API_URL = "http://localhost:6000"

def get_embeddings(texts):
    """Get embeddings for a list of texts."""
    response = requests.post(
        f"{API_URL}/embed",
        json={"texts": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    # Example 1: Simple embedding
    print("Example 1: Simple embedding")
    text = "Norge er et vakkert land med fjorder og fjell."
    embeddings = get_embeddings([text])
    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print()
    
    # Example 2: Semantic search
    print("Example 2: Semantic search")
    
    # Documents to search
    documents = [
        "Python er et populært programmeringsspråk for datavitenskap.",
        "Norge har mange vakre fjorder langs kysten.",
        "Machine learning brukes ofte i moderne applikasjoner.",
        "Fjellene i Norge er perfekte for skiing om vinteren.",
        "Kunstig intelligens endrer hvordan vi arbeider."
    ]
    
    # Query
    query = "Programmering og AI"
    
    # Get embeddings
    doc_embeddings = get_embeddings(documents)
    query_embedding = get_embeddings([query])[0]
    
    # Calculate similarities
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((sim, i))
    
    # Sort by similarity
    similarities.sort(reverse=True)
    
    print(f"Query: '{query}'")
    print("\nTop matching documents:")
    for sim, idx in similarities[:3]:
        print(f"  Score: {sim:.3f} - {documents[idx]}")
    print()
    
    # Example 3: Clustering similar texts
    print("Example 3: Finding similar texts")
    
    texts = [
        "Jeg liker å gå tur i skogen.",
        "Turgåing i naturen er avslappende.",
        "Fotball er Norges mest populære sport.",
        "Mange nordmenn elsker å se fotball.",
        "Programmering krever logisk tenkning.",
        "Koding er en viktig ferdighet i dag."
    ]
    
    embeddings = get_embeddings(texts)
    
    # Find most similar pairs
    print("\nSimilar text pairs:")
    threshold = 0.8
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > threshold:
                print(f"  Similarity {sim:.3f}:")
                print(f"    - {texts[i]}")
                print(f"    - {texts[j]}")

if __name__ == "__main__":
    main()
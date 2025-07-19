#!/bin/bash
# Examples of using the Nordic Embedding Service with curl

API_URL="http://localhost:6000"

echo "=== Nordic Embedding Service - curl examples ==="
echo

# Example 1: Health check
echo "1. Health check:"
curl -s "$API_URL/health" | python3 -m json.tool
echo

# Example 2: Service info
echo "2. Service info:"
curl -s "$API_URL/api/info" | python3 -m json.tool
echo

# Example 3: Single text embedding
echo "3. Single text embedding:"
curl -s -X POST "$API_URL/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hei, hvordan har du det?"]
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
emb = data['embeddings'][0]
print(f'Embedding size: {len(emb)}')
print(f'First 5 dimensions: {emb[:5]}')
"
echo

# Example 4: Batch embedding
echo "4. Batch embedding:"
curl -s -X POST "$API_URL/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Oslo er hovedstaden i Norge.",
      "Bergen er kjent for regn.",
      "Trondheim har Nidarosdomen."
    ]
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
embeddings = data['embeddings']
print(f'Number of embeddings: {len(embeddings)}')
for i, emb in enumerate(embeddings):
    print(f'  Text {i+1}: {len(emb)} dimensions')
"
echo

# Example 5: Similarity calculation
echo "5. Semantic similarity example:"
echo "Comparing similar and dissimilar Norwegian sentences..."

# Create temporary Python script for similarity calculation
cat << 'EOF' > /tmp/similarity.py
import sys, json, requests
import numpy as np

api_url = sys.argv[1]

texts = [
    "Jeg elsker å spise pizza.",
    "Pizza er min favorittrett.",
    "Fotball er gøy å spille.",
    "Jeg liker å svømme i havet."
]

# Get embeddings
response = requests.post(f"{api_url}/embed", json={"texts": texts})
embeddings = response.json()["embeddings"]

# Calculate cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Texts:")
for i, text in enumerate(texts):
    print(f"  {i}: {text}")

print("\nSimilarity matrix:")
print("     ", end="")
for i in range(len(texts)):
    print(f"{i:>6}", end="")
print()

for i in range(len(texts)):
    print(f"{i:>3}: ", end="")
    for j in range(len(texts)):
        sim = cosine_sim(embeddings[i], embeddings[j])
        print(f"{sim:>6.3f}", end="")
    print()

print("\nInterpretation:")
print("- Texts 0 and 1 (about pizza) should have high similarity")
print("- Other pairs should have lower similarity")
EOF

python3 /tmp/similarity.py "$API_URL"
rm /tmp/similarity.py
echo

# Example 6: Performance test
echo "6. Performance test (10 texts):"
TIME_START=$(date +%s.%N)

curl -s -X POST "$API_URL/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Dette er test nummer en.",
      "Dette er test nummer to.",
      "Dette er test nummer tre.",
      "Dette er test nummer fire.",
      "Dette er test nummer fem.",
      "Dette er test nummer seks.",
      "Dette er test nummer syv.",
      "Dette er test nummer åtte.",
      "Dette er test nummer ni.",
      "Dette er test nummer ti."
    ]
  }' > /dev/null

TIME_END=$(date +%s.%N)
TIME_DIFF=$(echo "$TIME_END - $TIME_START" | bc)

echo "Time taken for 10 embeddings: ${TIME_DIFF}s"
echo "Average time per embedding: $(echo "scale=3; $TIME_DIFF / 10" | bc)s"
echo

echo "=== Examples completed ==="
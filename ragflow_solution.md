# RAGFlow Integration Solution

## The Problem

RAGFlow returns error "102 OpenAI-API-Compatible dose not support this model(nordic-reranker)" because:

1. RAGFlow expects all models to work through the `/v1/embeddings` endpoint
2. RAGFlow doesn't understand the distinction between embedding and reranking models
3. The error message has a typo ("dose" instead of "does"), indicating it's from RAGFlow, not our service

## Solutions

### Solution 1: Use Different Model Names in RAGFlow

Instead of using reranking model names directly, configure RAGFlow with embedding models only:

**For Embeddings:**
- Use: `norbert2`, `nb-bert-base`, `multilingual-e5-base`, etc.
- These work with `/v1/embeddings` endpoint

**For Reranking:**
- RAGFlow may need to be configured differently for reranking
- Check RAGFlow's documentation for "rerank_model" configuration
- Some RAGFlow versions don't support external reranking models via OpenAI API

### Solution 2: Configure RAGFlow Properly

In RAGFlow's configuration, you may need to:

1. **Set up embedding model separately from reranking model**
   ```yaml
   embedding:
     model_name: "norbert2"
     api_base: "http://your-server:7000/v1"
   
   reranking:
     enabled: false  # Or use RAGFlow's built-in reranking
   ```

2. **Use only embedding models**
   - Configure RAGFlow to use only embedding models
   - Let RAGFlow handle reranking internally using vector similarity

### Solution 3: Check RAGFlow Version

Different RAGFlow versions handle models differently:

1. **Older versions**: May only support embedding models via OpenAI API
2. **Newer versions**: May have separate configuration for reranking models

### Recommended Approach

1. **For now, use only embedding models with RAGFlow:**
   ```
   Model: norbert2 (or any other embedding model)
   Base URL: http://your-server:7000/v1
   ```

2. **For reranking**, you have options:
   - Use RAGFlow's built-in reranking (vector similarity)
   - Configure reranking separately if RAGFlow supports it
   - Use our reranking API directly in your application (bypass RAGFlow)

### Testing

Test that embedding models work:
```bash
# This should work with RAGFlow
curl -X POST http://your-server:7000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "norbert2",
    "input": "Test text"
  }'
```

### Direct Reranking Usage (Outside RAGFlow)

If you need reranking, use our API directly:
```python
import requests

# Get embeddings via RAGFlow
embeddings = ragflow_client.get_embeddings(texts)

# Use our reranking API directly
response = requests.post("http://your-server:7000/v1/rerank", json={
    "model": "mmarco-minilm-l12",
    "query": query,
    "documents": documents,
    "top_n": 10
})
reranked = response.json()
```

## Summary

The error occurs because RAGFlow is trying to use a reranking model (`nordic-reranker`) through the embeddings API. RAGFlow may not fully support external reranking models through OpenAI-compatible APIs. Use embedding models with RAGFlow and handle reranking separately if needed.
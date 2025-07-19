# Complete RAGFlow Configuration Guide for NoEmbed

## Overview

RAGFlow handles embedding and reranking models differently:
- **Embedding models**: Configured as OpenAI-compatible models
- **Reranking models**: Configured in `llm_factories.json` with special endpoints

## Part 1: Embedding Models Configuration

### Step 1: Add OpenAI-Compatible Provider in RAGFlow

1. Go to RAGFlow Model Providers
2. Click "OpenAI-API-Compatible"
3. Configure:
   ```
   Base URL: http://your-server:7000/v1
   API Key: any-value (required but not used)
   ```

### Step 2: Select Embedding Model

When creating a Knowledge Base, select one of these embedding models:
- `norbert2` - Norwegian BERT v2
- `nb-bert-base` - Norwegian BERT base
- `multilingual-e5-base` - Best for multilingual content
- `kb-bert-swedish` - Swedish BERT
- `dabert` - Danish BERT
- `finbert-base` - Finnish BERT
- `icebert` - Icelandic BERT

## Part 2: Reranking Models Configuration

### Step 1: Edit llm_factories.json

Add this to your `conf/llm_factories.json`:

```json
{
  "factory_llm_infos": [
    {
      "name": "NoEmbedReranker",
      "tags": "RE-RANK",
      "status": "1",
      "llm": [
        {
          "llm_name": "nordic-reranker@noembed",
          "tags": "RE-RANK",
          "model_type": "rerank",
          "base_url": "http://your-server:7000/rerank",
          "max_tokens": 512
        },
        {
          "llm_name": "mmarco-minilm-l12@noembed",
          "tags": "RE-RANK",
          "model_type": "rerank",
          "base_url": "http://your-server:7000/rerank",
          "max_tokens": 512
        }
      ]
    }
  ]
}
```

### Step 2: Restart RAGFlow

```bash
docker compose restart ragflow-server
```

### Step 3: Configure in Agent Workflow

1. Open your Agent workflow
2. Add/Edit Retrieval component
3. Set:
   - Top N: 6-10 (number of final results)
   - Keywords similarity weight: 0.3
   - Similarity threshold: 0.2
   - **Rerank model**: Select "nordic-reranker@noembed"

## Complete Example Configuration

### 1. For Norwegian Content

**Knowledge Base Settings:**
```
Embedding Model: norbert2
Chunk Size: 512
```

**Agent Retrieval Settings:**
```json
{
  "similarity_threshold": 0.2,
  "keywords_similarity_weight": 0.3,
  "top_n": 6,
  "top_k": 1024,
  "rerank_id": "nordic-reranker@noembed",
  "kb_ids": ["your-norwegian-kb-id"]
}
```

### 2. For Multilingual Content

**Knowledge Base Settings:**
```
Embedding Model: multilingual-e5-base
Chunk Size: 512
```

**Agent Retrieval Settings:**
```json
{
  "similarity_threshold": 0.2,
  "keywords_similarity_weight": 0.3,
  "top_n": 6,
  "top_k": 1024,
  "rerank_id": "mmarco-minilm-l12@noembed",
  "kb_ids": ["your-multilingual-kb-id"]
}
```

## Testing Your Configuration

### Test Embedding Model
```bash
curl -X POST http://your-server:7000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "norbert2",
    "input": "Test embedding"
  }'
```

### Test Reranking Model
```bash
curl -X POST http://your-server:7000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of Norway?",
    "docs": ["Oslo is the capital", "Bergen is a city"]
  }'
```

Expected response:
```json
{
  "scores": [0.9876, 0.1234]
}
```

## Common Issues and Solutions

### Issue: "OpenAI-API-Compatible does not support this model"

**Cause**: Trying to use a reranking model as an embedding model
**Solution**: Only use embedding models in Knowledge Base configuration

### Issue: Reranking model not showing in Agent UI

**Cause**: llm_factories.json not properly configured
**Solution**: 
1. Check JSON syntax in llm_factories.json
2. Ensure "model_type": "rerank" is set
3. Restart RAGFlow server

### Issue: Slow response times

**Solution**: 
1. Use smaller reranking model (ms-marco-minilm-l6)
2. Reduce top_k value (e.g., from 1024 to 512)
3. Enable GPU in NoEmbed (.env: DEVICE=cuda)

## Model Selection Guide

### For Norwegian/Scandinavian Content
- Embedding: `norbert2` or `nb-bert-base`
- Reranking: `nordic-reranker@noembed` (alias for mmarco-minilm-l12)

### For Multilingual Content
- Embedding: `multilingual-e5-base`
- Reranking: `mmarco-minilm-l12@noembed`

### For Speed Optimization
- Embedding: `electra-small-nordic` (fastest)
- Reranking: Configure without reranking (use vector similarity only)

## Summary

1. **Embeddings**: Use OpenAI-compatible configuration with `/v1` base URL
2. **Reranking**: Configure in `llm_factories.json` with `/rerank` endpoint
3. **Model names matter**: Use exact model names as listed
4. **Restart required**: After editing llm_factories.json

This separation allows RAGFlow to properly route requests to the correct endpoints and use the full capabilities of both embedding and reranking models.
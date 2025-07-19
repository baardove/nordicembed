# RAGFlow Integration Guide for NoEmbed

This guide explains how to integrate NoEmbed with RAGFlow for both embedding and reranking models.

## Overview

NoEmbed provides both embedding and reranking models through OpenAI-compatible APIs. However, RAGFlow has specific requirements for how these models should be configured.

## Model Types

NoEmbed supports two types of models:
- **Embedding Models**: Convert text to vector representations (23 models)
- **Reranking Models**: Score relevance between queries and documents (5 models)

## RAGFlow Configuration

### 1. Add NoEmbed as OpenAI-Compatible Provider

In RAGFlow, go to **Model Providers** and click **OpenAI-API-Compatible**, then configure:

```
Base URL: http://your-server:7000/v1
API Key: any-value (not used but required by RAGFlow)
```

### 2. Configure Embedding Models

For embedding models, use the standard configuration:

```json
{
  "model_type": "embedding",
  "model_name": "norbert2",  // or any embedding model
  "api_base": "http://your-server:7000/v1",
  "api_key": "any-value"
}
```

Available embedding models:
- Norwegian: `norbert2`, `nb-bert-base`, `nb-bert-large`, etc.
- Swedish: `kb-bert-swedish`, `bert-large-swedish`, etc.
- Danish: `dabert`, `aelaectra-danish`, etc.
- Finnish: `finbert-base`, `finbert-sbert`, etc.
- Icelandic: `icebert`
- Multilingual: `xlm-roberta-base`, `multilingual-e5-base`, etc.

### 3. Configure Reranking Models

RAGFlow may require special configuration for reranking models. Try these approaches:

#### Option A: Direct Reranking Configuration
```json
{
  "model_type": "reranking",
  "model_name": "mmarco-minilm-l12",
  "api_base": "http://your-server:7000/api/v1",
  "api_key": "any-value"
}
```

#### Option B: As Custom Model
```json
{
  "model_type": "custom",
  "model_name": "mmarco-minilm-l12",
  "api_base": "http://your-server:7000/v1",
  "api_key": "any-value",
  "endpoint_type": "rerank"
}
```

Available reranking models:
- `mmarco-minilm-l12` (Recommended for Nordic languages)
- `ms-marco-minilm-l6` (Fast, English-focused)
- `ms-marco-minilm-l12` (High quality, English-focused)
- `jina-reranker-multilingual` (Modern multilingual)
- `nordic-reranker` (Alias for mmarco-minilm-l12)

## Troubleshooting

### Error: "Model is a reranking model"

This error occurs when you try to use a reranking model with the embeddings endpoint.

**Solution**: Configure the model as a reranking model in RAGFlow, not as an embedding model.

### Error: "OpenAI-API-Compatible does not support this model"

This can happen if:
1. The model name is misspelled
2. You're using a reranking model as an embedding model
3. RAGFlow is not configured correctly

**Solution**: 
1. Check model name spelling
2. Verify model type using: `python check_model_type.py <model_name>`
3. Use the correct endpoint configuration in RAGFlow

### Checking Model Availability

List all available models:
```bash
curl http://your-server:7000/v1/models | jq '.data[] | {id, type}'
```

Check specific model type:
```bash
python check_model_type.py mmarco-minilm-l12
```

## Testing

### Test Embedding Model
```bash
curl -X POST http://your-server:7000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "norbert2",
    "input": "Test text"
  }'
```

### Test Reranking Model
```bash
# Via standard endpoint
curl -X POST http://your-server:7000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mmarco-minilm-l12",
    "query": "Norwegian capital",
    "documents": ["Oslo is the capital", "Bergen is a city"]
  }'

# Via RAGFlow-compatible endpoint
curl -X POST http://your-server:7000/api/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mmarco-minilm-l12",
    "query": "Norwegian capital",
    "documents": ["Oslo is the capital", "Bergen is a city"]
  }'
```

## Best Practices

1. **Use appropriate models**: 
   - For Norwegian/Nordic text: Use Nordic-specific models
   - For multilingual: Use `multilingual-e5-base` or `xlm-roberta-base`
   - For reranking: Use `mmarco-minilm-l12` for Nordic languages

2. **Performance optimization**:
   - Start with smaller models (`electra-small-nordic`, `albert-swedish`)
   - Use GPU if available (set `DEVICE=cuda` in `.env`)
   - Batch requests when possible

3. **Model loading**:
   - Models are downloaded automatically on first use
   - Pre-download for production: `python download_models.py <model_name>`

## Support

If you encounter issues:
1. Check the logs: `docker logs noembed-container`
2. Verify model availability: `curl http://your-server:7000/v1/models`
3. Test endpoints directly using the curl commands above
4. Use `check_model_type.py` to verify model types
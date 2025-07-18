# RAGFlow Integration Guide

## Prerequisites
- RAGFlow instance running
- Norwegian Embedding Service running on port 6000

## Integration Steps

1. **Start the Embedding Service**
   ```bash
   docker-compose up -d embed-norwegian
   ```

2. **Copy Configuration to RAGFlow**
   Copy the `llm_factories.json` content to your RAGFlow configuration:
   ```bash
   # Append to existing RAGFlow conf/llm_factories.json
   cat ragflow/llm_factories.json >> /path/to/ragflow/conf/llm_factories.json
   ```

3. **Restart RAGFlow**
   ```bash
   docker-compose restart ragflow
   ```

4. **Configure in RAGFlow UI**
   - Go to Knowledge Base settings
   - Select "NorwegianEmbeddings" as the embedding model
   - Choose specific model variant:
     - `norbert2-embed@local` - For general Norwegian text
     - `simcse-nb-bert-embed@local` - For sentence similarity
     - `norbert3-base-embed@local` - For lighter workloads

## Network Configuration

If RAGFlow and the embedding service are in different Docker networks:

1. **Option 1: Use host network**
   Change `base_url` in llm_factories.json to use your host IP:
   ```json
   "base_url": "http://YOUR_HOST_IP:6000/embed"
   ```

2. **Option 2: Connect networks**
   ```bash
   docker network connect ragflow_default norwegian-embeddings
   ```

## Verification

Test the integration:
```bash
curl -X POST http://localhost:6000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Test norsk tekst"]}'
```

## Troubleshooting

- Check service logs: `docker-compose logs embed-norwegian`
- Verify network connectivity between containers
- Ensure models are downloaded: `./download_models.py --path ./models`
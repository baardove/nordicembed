Below is the comprehensive PRD for building a local embedding service that supports Norwegian/Scandinavian models (e.g., NorBERT-2, SimCSE NB-BERT-large) for use with RAGFlow.


⸻

📘 Product Requirements Document: Local Embedding Service for RAGFlow

1. 🎯 Objective

Provide a local API that computes text embeddings using Norwegian and Scandinavian models, compatible with RAGFlow. Avoid external APIs to maintain data privacy and operate fully offline.

⸻

2. ✅ Features Overview
	•	Support multiple embedding models:
	•	NorBERT‑2 (Norwegian, cased base, 221M parameters)
	•	FFI/SimCSE‑NB‑BERT‑large (sentence embeddings, Norwegian)
	•	NorBERT‑3 base (lighter base model)
	•	Expose a REST JSON API (/embed) for RAGFlow integration
	•	Stateless, scalable, can batch-process texts
	•	Containerized via Docker for easy deployment
	•	Clear configuration for model selection & scalability
	•	Add simple web dashboard for configuration & status 
    •   if possible also add to dashbaoard  easy upload or fetching missing models     

⸻

3. 📦 Model Details & Download URLs

Model	Description	Download URL
NorBERT‑2	Norwegian BERT; good for historical/modern text	https://huggingface.co/ltg/norbert2/resolve/main/221.zip  ￼ ￼ ￼ ￼
FFI/SimCSE‑NB‑BERT‑large	Contrastive sentence embeddings in Norwegian	https://huggingface.co/FFI/SimCSE-NB-BERT-large ()
NorBERT‑3 base	Updated lighter model	https://huggingface.co/ltg/norbert3-base ()


⸻

4. 🧑‍💻 Technical Design

4.1 Architecture
	•	FastAPI server exposing /embed, accepting JSON:

{ "texts": ["sentence A", "sentence B", ...] }


	•	Returns:

{ "embeddings": [[...], [...], ...] }


	•	Load models at startup based on configuration (environment variable: MODEL_NAME)

4.2 Performance
	•	Batch 16–32 texts per request
	•	Use GPU or CPU; use torch.no_grad() and pooled outputs
	•	Optional caching in memory

4.3 Containerization
	•	Dockerfile defines:
	•	Base image: python:3.10-slim
	•	Install transformers, torch, fastapi, uvicorn
	•	Expose port 8000
	•	Entrypoint reads MODEL_NAME and starts server

⸻

5. ⚙️ RAGFlow Integration

5.1 Register Service

Add to conf/llm_factories.json:

{
  "factory_llm_infos": [
    {
      "name": "NorwegianEmbeddings",
      "tags": "EMBEDDING",
      "status": "1",
      "llm": [
        {
          "llm_name": "norbert2-embed@local",
          "tags": "EMBEDDING",
          "model_type": "embedding",
          "base_url": "http://host.docker.internal:8000/embed"
        }
      ]
    }
  ]
}

5.2 Usage in UI
	•	In Knowledge Base setup, choose NorwegianEmbeddings
	•	RAGFlow sends chunks to /embed, receives back vectors

⸻

6. 🗂️ Project Plan & Milestones

Phase	Milestones
Requirements	Validate embedding/API format, finalize model downloads
MVP Dev	Implement server for /embed, test with one model (e.g., NorBERT‑2)
Dockerization	Create Dockerfile + Compose, test container locally
Multi-Model	Add FFI/SimCSE, NorBERT‑3 support with MODEL_NAME config
RAGFlow Integration	Update config file, test embedding end-to-end in RAGFlow UI
Optimization	Add batching, concurrency limits, GPU support check feature flag
Documentation	Write README with download URLs, setup steps, sample curl/tests
Delivery	Package service, provide integration steps, PR to RAGFlow repo


⸻

7. 🎯 Example Workflow
	1.	Download models:

wget https://huggingface.co/ltg/norbert2/resolve/main/221.zip
unzip 221.zip -d models/norbert2


	2.	Build Docker image:

docker build -t embed-norwegian .


	3.	Run embedding service:

docker run -d -e MODEL_NAME=norbert2 -p 6000:6y000 embed-norwegian


	4.	Register in RAGFlow and restart server
	5.	In RAGFlow UI, create KB → select NorwegianEmbeddings → parse docs

⸻

8. ✅ Success Metrics
	•	Embeddings computed within <500 ms per chunk
	•	RAGFlow successfully indexes and retrieves Norwegian text
	•	Multi-model support verified (NorBERT‑2 & SimCSE)
	•	Full documentation and config guidance provided

⸻

9. 📚 References
	•	NorBERT‑2 model & download info  ￼ ￼ ￼ ￼
	•	FFI/SimCSE‑NB‑BERT‑large  ￼
	•	NorBERT‑3 base availability  ￼

⸻

Let me know if you’d like the Dockerfile template, configuration files, or even a sample curl test script to get started!
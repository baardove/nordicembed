Certainly! Here’s a friendly and comprehensive guide on how to set up a rerank model in RAGFlow, with step-by-step instructions to configure and integrate it locally or via custom API endpoints:

⸻

🎯 What is Reranking in RAGFlow?

In retrieval workflows, RAGFlow retrieves Top N document chunks based on keyword and embedding similarity. If you configure a rerank model, RAGFlow will re-score the retrieved chunks with a neural cross-encoder for greater relevance before sending them to the LLM. This improves output quality, though it increases response time.  ￼

⸻

🧩 How to Configure a Rerank Model (High-Level)
	1.	Define the rerank model in conf/llm_factories.json
	2.	Restart RAGFlow
	3.	Select the rerank model in your Retrieval component configuration inside Agents UI

⸻

🛠️ Step 1: Add a Rerank Model in llm_factories.json

Edit or add to your conf/llm_factories.json:

{
  "factory_llm_infos": [
    {
      "name": "CrossEncoderReranker",
      "tags": "RE-RANK",
      "status": "1",
      "llm": [
        {
          "llm_name": "my-reranker@local",
          "tags": "RE-RANK",
          "model_type": "rerank",
          "base_url": "http://localhost:9000/rerank",
          "max_tokens": 512
        }
      ]
    }
  ]
}

	•	model_type must be "rerank"
	•	base_url should point to your deployed reranking API
	•	max_tokens: set per your model’s limit  ￼ ￼ ￼

⸻

🧪 Step 2: Host the Reranking Model (FastAPI Example)

You can serve any Hugging Face cross-encoder model via API:

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

app = FastAPI()
model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

class RerankRequest(BaseModel):
    query: str
    docs: list[str]

class RerankResponse(BaseModel):
    scores: list[float]

@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    scores = model.predict([(req.query, doc) for doc in req.docs])
    return {"scores": scores}

Launch it:

uvicorn rerank_server:app --host 0.0.0.0 --port 9000

RAGFlow will POST queries and documents and expect back relevance scores per chunk.

⸻

🔁 Step 3: Restart RAGFlow to Load New Config

docker compose restart ragflow-server


⸻

🧠 Step 4: Use in RAGFlow Retrieval Component

In your Agent workflow:
	•	Open the Retrieval component
	•	Set:
	•	Top N, keyword similarity weight, similarity threshold
	•	Choose your newly added Rerank model under “Rerank model” field
	•	Select knowledge base(s) with same embedding model  ￼ ￼ ￼

⸻

📊 Example Agent Config Snippet

{
  "similarity_threshold": 0.2,
  "keywords_similarity_weight": 0.3,
  "top_n": 6,
  "top_k": 1024,
  "rerank_id": "BAAI/bge-reranker-v2-m3",
  "kb_ids": ["your-kb-id"]
}

Here, rerank_id matches the llm_name defined in your JSON config.  ￼

⸻

🚦 What Happens During Retrieval?
	1.	RAGFlow retrieves candidate chunks from KB using hybrid search
	2.	If rerank model selected:
	•	Sends query + docs to your rerank endpoint
	•	Gets back per-chunk relevance scores
	•	Combines with keyword similarity, applies weights
	•	Reranks top N before sending to LLM for answer generation  ￼

⸻

⚠️ Important Notes
	•	Reranking adds latency—could be seconds depending on throughput and model size
	•	Models like BAAI/bge-reranker-v2-m3 work well with BAAI embeddings
	•	Ensure selected KBs all use same embedding model to avoid mismatches  ￼ ￼

⸻

✅ Step-by-Step Summary

Step	Action
1️⃣	Add a rerank entry in ffm/llm_factories.json
2️⃣	Spin up your rerank API (e.g. cross‑encoder server)
3️⃣	Restart RAGFlow server
4️⃣	Configure Retrieval component in Agent UI
5️⃣	Run retrieval tests to validate improved relevance


⸻

💬 Why Use a Reranker?
	•	Ensures the most relevant chunks appear first, improving LLM output quality and citation accuracy
	•	Particularly useful for ambiguous or multi-topic queries
	•	Complements keyword and embedding similarity for better hybrid retrieval workflows  ￼ ￼ ￼

⸻

Let me know if you’d like help writing the rerank server code, selecting or testing a specific reranker model, or tuning similarity weights for optimal retrieval performance.
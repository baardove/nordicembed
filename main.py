import os
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from embeddings import EmbeddingService
from reranker import RerankerService
from config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()
embedding_service = None

# Metrics tracking
class MetricsTracker:
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.last_request_time = None
        self.last_request_info = {}
        self.response_times = []
        self.start_time = datetime.now()
        self.request_history = []  # Store (timestamp, duration) tuples
        self.model_requests = {}  # Track requests per model
        self.model_response_times = {}  # Track response times per model (excluding downloads)
        
    def track_request(self, endpoint: str, text_count: int, response_time: float, source_ip: str = None, model: str = None):
        self.total_requests += 1
        self.last_request_time = datetime.now()
        self.response_times.append(response_time)
        # Keep only last 100 response times
        if len(self.response_times) > 100:
            self.response_times.pop(0)
        
        # Track request history for utilization calculation
        self.request_history.append((self.last_request_time, response_time))
        # Clean up old entries (older than 10 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.request_history = [(ts, dur) for ts, dur in self.request_history if ts > cutoff_time]
        
        # Track model-specific requests and response times
        if model:
            if model not in self.model_requests:
                self.model_requests[model] = 0
                self.model_response_times[model] = []
            self.model_requests[model] += 1
            # Only track response times after first request (to exclude download time)
            if self.model_requests[model] > 1:
                self.model_response_times[model].append(response_time)
                # Keep only last 20 response times per model
                if len(self.model_response_times[model]) > 20:
                    self.model_response_times[model].pop(0)
        
        self.last_request_info = {
            "endpoint": endpoint,
            "text_count": text_count,
            "response_time": response_time,
            "timestamp": self.last_request_time.isoformat(),
            "source_ip": source_ip,
            "model": model
        }
    
    def get_metrics(self):
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        # Calculate utilization for last 10 minutes
        now = datetime.now()
        ten_minutes_ago = now - timedelta(minutes=10)
        window_seconds = 600  # 10 minutes in seconds
        
        # Calculate total busy time in the last 10 minutes
        busy_time = 0
        for timestamp, duration in self.request_history:
            if timestamp > ten_minutes_ago:
                busy_time += duration
        
        # Calculate utilization percentage (0-100)
        utilization = min((busy_time / window_seconds) * 100, 100) if window_seconds > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_response_time": round(avg_response_time * 1000, 2),  # in ms
            "last_request": self.last_request_info,
            "uptime_seconds": int(uptime_seconds),
            "requests_per_minute": round(self.total_requests / (uptime_seconds / 60), 2) if uptime_seconds > 0 else 0,
            "utilization_percentage": round(utilization, 1),
            "busy_seconds_10min": round(busy_time, 2),
            "model_requests": self.model_requests,
            "model_response_times": {
                model: {
                    "avg": round(sum(times) * 1000 / len(times), 1) if times else None,  # Convert to ms
                    "count": len(times)
                }
                for model, times in self.model_response_times.items()
            }
        }

metrics = MetricsTracker()


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    pooling_strategy: str = Field(default="mean", description="Pooling strategy: mean, max, or cls")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")


class HuggingFaceRequest(BaseModel):
    inputs: str | List[str] = Field(..., description="Text(s) to embed")


class HuggingFaceResponse(BaseModel):
    embeddings: List[List[float]]


class OllamaRequest(BaseModel):
    model: str
    prompt: str


class OllamaResponse(BaseModel):
    embedding: List[float]


class OpenAIEmbeddingRequest(BaseModel):
    input: str | List[str] = Field(..., description="Text(s) to embed")
    model: str = Field(default="text-embedding-ada-002", description="Model name (ignored)")


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_service
    logger.info(f"Loading model: {settings.model_name}")
    embedding_service = EmbeddingService(
        model_name=settings.model_name,
        model_path=settings.model_path,
        device=settings.device,
        max_length=settings.max_length
    )
    yield
    logger.info("Shutting down embedding service")


app = FastAPI(
    title="Norwegian Embedding Service",
    description="Local embedding service for Norwegian/Scandinavian language models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": embedding_service is not None
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint"""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": 1677532384,
                "owned_by": "norwegian-embeddings",
                "permission": [],
                "root": settings.model_name,
                "parent": None,
            }
        ]
    }


@app.get("/api/info")
async def service_info():
    return {
        "service": "Norwegian Embedding Service",
        "model": settings.model_name,
        "device": settings.device,
        "max_batch_size": settings.max_batch_size,
        "max_length": settings.max_length,
        "available_models": [
            # Norwegian models
            "norbert2", "nb-bert-base", "nb-bert-large", 
            "simcse-nb-bert-large", "norbert3-base", "norbert3-large",
            "xlm-roberta-base", "electra-small-nordic", "sentence-bert-base",
            # Swedish models
            "kb-sbert-swedish", "kb-bert-swedish", "bert-large-swedish", "albert-swedish",
            # Danish models
            "dabert", "aelaectra-danish", "da-bert-ner", "electra-base-danish",
            # Multilingual Nordic
            "multilingual-e5-base", "paraphrase-multilingual-minilm",
            # Finnish models
            "finbert-base", "finbert-sbert", "finbert-large",
            # Icelandic model
            "icebert"
        ],
        "available_rerankers": list(RerankerService.RERANKER_CONFIGS.keys()),
        "status": "running"
    }


@app.get("/api/metrics")
async def get_metrics():
    return metrics.get_metrics()


@app.post("/api/embed", response_model=EmbedResponse)
async def generate_embeddings(request: EmbedRequest):
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}"
        )
    
    try:
        start_time = time.time()
        # Pass pooling strategy to embedding service if supported
        embeddings = embedding_service.embed(request.texts)
        response_time = time.time() - start_time
        
        # Track metrics
        metrics.track_request("/api/embed", len(request.texts), response_time, model=settings.model_name)
        
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_compatible_embeddings(request: OpenAIEmbeddingRequest):
    """OpenAI-compatible embeddings endpoint for RAGFlow and other tools"""
    logger.info(f"Received embedding request - Input type: {type(request.input)}, Length: {len(request.input) if isinstance(request.input, list) else 1}")
    
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract model name from request (default to current model)
    requested_model = request.model if request.model else settings.model_name
    
    # Convert input to list if it's a string
    texts = [request.input] if isinstance(request.input, str) else request.input
    
    if not texts:
        raise HTTPException(status_code=400, detail="No input provided")
    
    if len(texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}"
        )
    
    try:
        start_time = time.time()
        embeddings = embedding_service.embed(texts)
        embed_time = time.time() - start_time
        logger.info(f"Generated embeddings for {len(texts)} texts in {embed_time:.3f}s")
        
        # Track metrics
        metrics.track_request("/api/v1/embeddings", len(texts), embed_time, model=requested_model)
        
        # Format response in OpenAI format
        data = []
        for i, embedding in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "embedding": embedding,
                "index": i
            })
        
        return OpenAIEmbeddingResponse(
            object="list",
            data=data,
            model=settings.model_name,
            usage={
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=str(e))


class TestEmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")
    model: str = Field(..., description="Model to use for embeddings")
    pooling_strategy: str = Field(default="mean", description="Pooling strategy: mean, max, or cls")


@app.post("/api/test-embed", response_model=EmbedResponse)
async def test_embeddings(request: TestEmbedRequest):
    """Test endpoint that allows specifying which model to use"""
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}"
        )
    
    # Check if requested model is available
    if request.model not in embedding_service.MODEL_CONFIGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model}' not available. Available models: {list(embedding_service.MODEL_CONFIGS.keys())}"
        )
    
    try:
        start_time = time.time()
        
        # Create a temporary embedding service with the requested model
        from embeddings import EmbeddingService as ES
        test_service = ES(
            model_name=request.model,
            model_path=settings.model_path,
            device=settings.device,
            max_length=settings.max_length
        )
        
        # Generate embeddings
        embeddings = test_service.embed(request.texts)
        response_time = time.time() - start_time
        
        # Track metrics
        metrics.track_request("/api/test-embed", len(request.texts), response_time, model=request.model)
        
        # Clean up
        del test_service
        
        return EmbedResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error generating test embeddings: {str(e)}")
        metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=str(e))


class RerankRequest(BaseModel):
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="List of documents to rerank")
    model: str = Field(default="mmarco-minilm-l12", description="Reranker model to use")
    top_k: int = Field(default=None, description="Return only top K documents")


class RerankResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="Reranked documents with scores")
    model: str = Field(..., description="Model used for reranking")
    query_length: int = Field(..., description="Length of the query")
    documents_count: int = Field(..., description="Number of documents reranked")


class ScorePairsRequest(BaseModel):
    pairs: List[List[str]] = Field(..., description="List of [query, document] pairs to score")
    model: str = Field(default="mmarco-minilm-l12", description="Reranker model to use")


class ScorePairsResponse(BaseModel):
    scores: List[float] = Field(..., description="Relevance scores for each pair")
    model: str = Field(..., description="Model used for scoring")


class DeviceUpdateRequest(BaseModel):
    device: str = Field(..., description="Device to use: cpu or cuda")


@app.post("/api/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on relevance to query using cross-encoder models"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    # Check if requested model is available
    if request.model not in RerankerService.RERANKER_CONFIGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Reranker model '{request.model}' not available. Available models: {list(RerankerService.RERANKER_CONFIGS.keys())}"
        )
    
    try:
        start_time = time.time()
        
        # Create reranker service
        reranker = RerankerService(
            model_name=request.model,
            model_path=settings.model_path,
            device=settings.device,
            max_length=settings.max_length
        )
        
        # Rerank documents
        results = reranker.rerank(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k
        )
        
        response_time = time.time() - start_time
        
        # Track metrics
        metrics.track_request("/api/rerank", len(request.documents), response_time, model=request.model)
        
        # Format results
        formatted_results = [
            {
                "index": idx,
                "score": score,
                "document": doc
            }
            for idx, score, doc in results
        ]
        
        # Clean up
        del reranker
        
        return RerankResponse(
            results=formatted_results,
            model=request.model,
            query_length=len(request.query),
            documents_count=len(request.documents)
        )
    except Exception as e:
        logger.error(f"Error reranking documents: {str(e)}")
        metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/score-pairs", response_model=ScorePairsResponse)
async def score_pairs(request: ScorePairsRequest):
    """Score query-document pairs using cross-encoder models"""
    if not request.pairs:
        raise HTTPException(status_code=400, detail="No pairs provided")
    
    # Validate pairs format
    for pair in request.pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise HTTPException(status_code=400, detail="Each pair must be a list of [query, document]")
    
    # Check if requested model is available
    if request.model not in RerankerService.RERANKER_CONFIGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Reranker model '{request.model}' not available. Available models: {list(RerankerService.RERANKER_CONFIGS.keys())}"
        )
    
    try:
        start_time = time.time()
        
        # Create reranker service
        reranker = RerankerService(
            model_name=request.model,
            model_path=settings.model_path,
            device=settings.device,
            max_length=settings.max_length
        )
        
        # Score pairs
        scores = reranker.score_pairs([(pair[0], pair[1]) for pair in request.pairs])
        
        response_time = time.time() - start_time
        
        # Track metrics
        metrics.track_request("/api/score-pairs", len(request.pairs), response_time, model=request.model)
        
        # Clean up
        del reranker
        
        return ScorePairsResponse(
            scores=scores,
            model=request.model
        )
    except Exception as e:
        logger.error(f"Error scoring pairs: {str(e)}")
        metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/update-device")
async def update_device_setting(request: DeviceUpdateRequest):
    """Update the device setting in the .env file"""
    if request.device not in ["cpu", "cuda"]:
        raise HTTPException(status_code=400, detail="Device must be 'cpu' or 'cuda'")
    
    try:
        # Read the current .env file
        env_path = ".env"
        lines = []
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
        
        # Update or add DEVICE setting
        device_found = False
        for i, line in enumerate(lines):
            if line.strip().startswith('DEVICE='):
                lines[i] = f"DEVICE={request.device}  # cpu or cuda\n"
                device_found = True
                break
        
        # If DEVICE not found, add it
        if not device_found:
            lines.append(f"\nDEVICE={request.device}  # cpu or cuda\n")
        
        # Write back to .env file
        with open(env_path, 'w') as f:
            f.writelines(lines)
        
        logger.info(f"Updated device setting to: {request.device}")
        
        return {
            "status": "success",
            "message": f"Device setting updated to {request.device}",
            "device": request.device,
            "note": "Container restart required for changes to take effect"
        }
        
    except Exception as e:
        logger.error(f"Error updating device setting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update device setting: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    )
import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Form, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
import uvicorn

from embeddings import EmbeddingService
from reranker import RerankerService
from config import get_settings, save_device_config
from auth_config import (
    load_auth_config, save_auth_config, verify_dashboard_password,
    verify_api_key, add_api_key, remove_api_key, get_api_key_stats,
    hash_password, get_or_create_internal_key
)
from ragflow_adapter import (
    is_ragflow_request, 
    convert_ragflow_to_standard_rerank,
    convert_standard_to_ragflow_response,
    RAGFlowRerankRequest,
    RAGFlowRerankResponse
)

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
    title="Nordic Embedding Service",
    description="Local embedding service for Nordic language models",
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


# Authentication middleware
@app.middleware("http")
async def auth_middleware(request, call_next):
    """Check API key authentication for protected endpoints"""
    path = request.url.path
    
    # Skip auth for dashboard, static files, and auth endpoints
    if path.startswith("/static") or path.startswith("/api/auth") or path in ["/", "/favicon.ico", "/api/health"]:
        return await call_next(request)
    
    # Check if API auth is enabled
    auth_config = load_auth_config()
    if auth_config.api_auth_enabled and path.startswith("/api"):
        # Extract API key from header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key and api_key.startswith("Bearer "):
            api_key = api_key[7:]  # Remove "Bearer " prefix
        
        # Verify API key
        key_name = verify_api_key(api_key) if api_key else None
        if not key_name:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"}
            )
        
        # Add key name to request state for logging
        request.state.api_key_name = key_name
    
    response = await call_next(request)
    return response


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve dashboard with optional password protection"""
    auth_config = load_auth_config()
    
    # Check if dashboard auth is enabled
    if auth_config.dashboard_auth_enabled:
        # Check for auth cookie
        auth_cookie = request.cookies.get("dashboard_auth")
        
        # If no valid auth cookie, show login form
        if not auth_cookie or auth_cookie != auth_config.dashboard_password_hash:
            return HTMLResponse(content="""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Nordic Embeddings - Login</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            display: flex; 
                            justify-content: center; 
                            align-items: center; 
                            height: 100vh; 
                            margin: 0;
                            background-color: #f5f5f5;
                        }
                        .login-box {
                            background: white;
                            padding: 40px;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                            width: 300px;
                        }
                        h2 { margin-bottom: 20px; color: #333; }
                        input[type="password"] {
                            width: 100%;
                            padding: 10px;
                            margin-bottom: 20px;
                            border: 1px solid #ddd;
                            border-radius: 4px;
                            font-size: 16px;
                        }
                        button {
                            width: 100%;
                            padding: 10px;
                            background-color: #3498db;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            font-size: 16px;
                            cursor: pointer;
                        }
                        button:hover { background-color: #2980b9; }
                        .error { color: #e74c3c; margin-top: 10px; }
                    </style>
                </head>
                <body>
                    <div class="login-box">
                        <h2>Nordic Embeddings Dashboard</h2>
                        <form method="post" action="/login">
                            <input type="password" name="password" placeholder="Enter password" required autofocus>
                            <button type="submit">Login</button>
                        </form>
                        <div class="error" id="error"></div>
                    </div>
                    <script>
                        if (window.location.search.includes('password=')) {
                            document.getElementById('error').textContent = 'Invalid password';
                        }
                    </script>
                </body>
                </html>
            """)
    
    # Serve dashboard
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/login")
async def login(response: Response, password: str = Form(...)):
    """Handle login form submission"""
    auth_config = load_auth_config()
    
    if verify_dashboard_password(password):
        # Set a secure cookie with the password hash
        response = RedirectResponse(url="/", status_code=303)
        response.set_cookie(
            key="dashboard_auth",
            value=hash_password(password),
            httponly=True,
            secure=False,  # Set to True if using HTTPS
            samesite="lax",
            max_age=86400  # 24 hours
        )
        return response
    else:
        # Return login form with error
        return HTMLResponse(content="""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Nordic Embeddings - Login</title>
                <style>
                    body { 
                        font-family: Arial, sans-serif; 
                        display: flex; 
                        justify-content: center; 
                        align-items: center; 
                        height: 100vh; 
                        margin: 0;
                        background-color: #f5f5f5;
                    }
                    .login-box {
                        background: white;
                        padding: 40px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        width: 300px;
                    }
                    h2 { margin-bottom: 20px; color: #333; }
                    input[type="password"] {
                        width: 100%;
                        padding: 10px;
                        margin-bottom: 20px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        font-size: 16px;
                    }
                    button {
                        width: 100%;
                        padding: 10px;
                        background-color: #3498db;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        font-size: 16px;
                        cursor: pointer;
                    }
                    button:hover { background-color: #2980b9; }
                    .error { 
                        color: #e74c3c; 
                        margin-top: 10px; 
                        text-align: center;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="login-box">
                    <h2>Nordic Embeddings Dashboard</h2>
                    <form method="post" action="/login">
                        <input type="password" name="password" placeholder="Enter password" required autofocus>
                        <button type="submit">Login</button>
                    </form>
                    <div class="error">Invalid password</div>
                </div>
            </body>
            </html>
        """, status_code=401)


@app.get("/logout")
async def logout():
    """Logout by clearing the auth cookie"""
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("dashboard_auth")
    return response


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import FileResponse
    return FileResponse("static/favicon.ico")


@app.get("/debug", response_class=HTMLResponse)
async def debug_dashboard():
    with open("debug_dashboard.html", "r") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": embedding_service is not None
    }


@app.get("/v1/models")
async def list_models():
    """List all available models (OpenAI-compatible)"""
    models = []
    
    # Add embedding models
    for model_id in EmbeddingService.MODEL_CONFIGS.keys():
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1700000000,  # Placeholder timestamp
            "owned_by": "noembed",
            "permission": [],
            "root": model_id,
            "parent": None,
            "capabilities": {
                "embeddings": True,
                "reranking": False
            },
            "type": "embedding"
        })
    
    # Add reranking models with clearer naming for RAGFlow
    for model_id in RerankerService.RERANKER_CONFIGS.keys():
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1700000000,  # Placeholder timestamp
            "owned_by": "noembed",
            "permission": [],
            "root": model_id,
            "parent": None,
            "capabilities": {
                "embeddings": False,
                "reranking": True
            },
            "type": "reranking"
        })
        
        # Also add with -reranker suffix for clarity
        if not model_id.endswith("-reranker"):
            models.append({
                "id": f"{model_id}-reranker",
                "object": "model",
                "created": 1700000000,
                "owned_by": "noembed",
                "permission": [],
                "root": model_id,
                "parent": None,
                "capabilities": {
                    "embeddings": False,
                    "reranking": True
                },
                "type": "reranking"
            })
    
    return {
        "object": "list",
        "data": models
    }


@app.get("/api/info")
async def service_info():
    return {
        "service": "Nordic Embedding Service",
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
        "allow_trust_remote_code": settings.allow_trust_remote_code,
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


@app.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_compatible_embeddings(request: OpenAIEmbeddingRequest):
    """OpenAI-compatible embeddings endpoint for RAGFlow and other tools"""
    logger.info(f"Received embedding request - Model: {request.model}, Input type: {type(request.input)}, Length: {len(request.input) if isinstance(request.input, list) else 1}")
    logger.info(f"Request headers: {request.__dict__ if hasattr(request, '__dict__') else 'N/A'}")
    
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract model name from request (default to current model)
    requested_model = request.model if request.model else settings.model_name
    
    # Check if this is a reranking model being requested through the wrong endpoint
    if requested_model in RerankerService.RERANKER_CONFIGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{requested_model}' is a reranking model. Use /v1/rerank endpoint instead of /v1/embeddings for reranking tasks."
        )
    
    # Check if the requested model is available for embeddings
    if requested_model not in embedding_service.MODEL_CONFIGS:
        # Check if it might be a reranking model (double check)
        if requested_model in RerankerService.RERANKER_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{requested_model}' is a reranking model. Please use POST /v1/rerank instead of /v1/embeddings."
            )
        raise HTTPException(
            status_code=400,
            detail=f"Model '{requested_model}' is not available. Available embedding models: {', '.join(sorted(embedding_service.MODEL_CONFIGS.keys()))}"
        )
    
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
        
        # If a different model is requested, create a temporary service
        if requested_model != settings.model_name:
            from embeddings import EmbeddingService as ES
            temp_service = ES(
                model_name=requested_model,
                model_path=settings.model_path,
                device=settings.device,
                max_length=settings.max_length
            )
            embeddings = temp_service.embed(texts)
            del temp_service
        else:
            embeddings = embedding_service.embed(texts)
        
        embed_time = time.time() - start_time
        logger.info(f"Generated embeddings for {len(texts)} texts in {embed_time:.3f}s using model {requested_model}")
        
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


class OpenAIRerankRequest(BaseModel):
    model: str = Field(..., description="Model to use for reranking")
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(..., description="List of documents to rerank")
    top_n: Optional[int] = Field(None, description="Number of top results to return")
    return_documents: Optional[bool] = Field(True, description="Whether to return full documents")


class OpenAIRerankResponse(BaseModel):
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Dict[str, int]


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


@app.post("/v1/rerank", response_model=OpenAIRerankResponse)
async def openai_compatible_rerank(request: OpenAIRerankRequest):
    """OpenAI-compatible reranking endpoint"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    # Check if requested model is available
    if request.model not in RerankerService.RERANKER_CONFIGS:
        # Check if it's an embedding model
        if request.model in EmbeddingService.MODEL_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' is an embedding model. Use /v1/embeddings endpoint for embeddings."
            )
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model}' not available. Available reranking models: {', '.join(RerankerService.RERANKER_CONFIGS.keys())}"
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
            top_k=request.top_n
        )
        
        response_time = time.time() - start_time
        
        # Track metrics
        metrics.track_request("/v1/rerank", len(request.documents), response_time, model=request.model)
        
        # Format response in OpenAI-compatible format
        data = []
        for idx, (original_idx, score, doc) in enumerate(results):
            result_item = {
                "index": idx,
                "relevance_score": float(score),
                "original_index": original_idx
            }
            if request.return_documents:
                result_item["document"] = doc
            data.append(result_item)
        
        # Clean up
        del reranker
        
        # Calculate token usage (approximate)
        total_chars = len(request.query) + sum(len(doc) for doc in request.documents)
        approx_tokens = total_chars // 4  # Rough approximation
        
        return OpenAIRerankResponse(
            object="list",
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": approx_tokens,
                "total_tokens": approx_tokens
            }
        )
    except Exception as e:
        logger.error(f"Error reranking documents: {str(e)}")
        metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank", response_model=RAGFlowRerankResponse)
async def ragflow_rerank_endpoint(request: RAGFlowRerankRequest):
    """RAGFlow-compatible reranking endpoint that matches their expected format"""
    logger.info(f"Received RAGFlow rerank request - Query: {request.query[:50]}...")
    
    # Determine which model to use based on URL path or default
    # RAGFlow will configure the model in llm_factories.json
    default_model = "mmarco-minilm-l12"  # Default reranking model
    
    try:
        start_time = time.time()
        
        # Create reranker service with default model
        reranker = RerankerService(
            model_name=default_model,
            model_path=settings.model_path,
            device=settings.device,
            max_length=settings.max_length
        )
        
        # Score each document against the query
        scores = reranker.score_pairs([(request.query, doc) for doc in request.docs])
        
        response_time = time.time() - start_time
        logger.info(f"Reranked {len(request.docs)} documents in {response_time:.3f}s")
        
        # Track metrics
        metrics.track_request("/rerank", len(request.docs), response_time, model=default_model)
        
        # Clean up
        del reranker
        
        # Return scores in RAGFlow expected format
        return RAGFlowRerankResponse(scores=scores)
        
    except Exception as e:
        logger.error(f"Error in RAGFlow reranking: {str(e)}")
        metrics.total_errors += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/update-device")
async def update_device_setting(request: DeviceUpdateRequest):
    """Update the device setting in the writable config directory"""
    if request.device.lower() not in ["cpu", "cuda"]:
        raise HTTPException(status_code=400, detail="Device must be 'cpu' or 'cuda'")
    
    try:
        # Save to writable config directory
        success = save_device_config(request.device)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save device configuration")
        
        logger.info(f"Updated device setting to: {request.device}")
        
        return {
            "status": "success",
            "message": f"Device setting updated to {request.device}",
            "device": request.device,
            "config_location": "config/device.json",
            "note": "Container restart required for changes to take effect"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating device setting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update device setting: {str(e)}")


@app.post("/api/config/trust-remote-code")
async def update_trust_remote_code(request: Dict[str, bool]):
    """Update the trust_remote_code setting"""
    enabled = request.get("enabled", True)
    
    try:
        # Save to .env file
        env_path = Path(".env")
        env_lines = []
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        
        # Update or add ALLOW_TRUST_REMOTE_CODE
        updated = False
        for i, line in enumerate(env_lines):
            if line.strip().startswith("ALLOW_TRUST_REMOTE_CODE="):
                env_lines[i] = f"ALLOW_TRUST_REMOTE_CODE={'true' if enabled else 'false'}\n"
                updated = True
                break
        
        if not updated:
            env_lines.append(f"\nALLOW_TRUST_REMOTE_CODE={'true' if enabled else 'false'}\n")
        
        with open(env_path, 'w') as f:
            f.writelines(env_lines)
        
        logger.info(f"Updated ALLOW_TRUST_REMOTE_CODE to: {enabled}")
        
        return {
            "status": "success",
            "message": f"Trust remote code {'enabled' if enabled else 'disabled'}",
            "enabled": enabled,
            "config_location": ".env",
            "note": "Container restart required for changes to take effect"
        }
    
    except Exception as e:
        logger.error(f"Error updating trust_remote_code setting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update setting: {str(e)}")


@app.post("/api/restart")
async def restart_container():
    """Restart the container (only works inside Docker)"""
    try:
        # Check if we're running in Docker
        if not os.path.exists("/.dockerenv"):
            raise HTTPException(
                status_code=503, 
                detail="Container restart only available when running in Docker"
            )
        
        # Send signal to restart (container should have restart policy)
        logger.info("Container restart requested")
        
        # Create a background task to exit after response
        import asyncio
        async def delayed_exit():
            await asyncio.sleep(1)
            os._exit(0)  # Force exit to trigger container restart
        
        asyncio.create_task(delayed_exit())
        
        return {
            "status": "success",
            "message": "Container restart initiated"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting container: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to restart: {str(e)}")


# Authentication endpoints
@app.get("/api/auth/status")
async def get_auth_status():
    """Get current authentication status"""
    config = load_auth_config()
    return {
        "dashboard_auth_enabled": config.dashboard_auth_enabled,
        "api_auth_enabled": config.api_auth_enabled,
        "api_keys_count": len(config.api_keys)
    }


@app.get("/api/auth/internal-key")
async def get_internal_key():
    """Get internal API key for system operations"""
    # This endpoint doesn't require auth since it's used by the dashboard
    # but only returns the key if API auth is enabled
    config = load_auth_config()
    if config.api_auth_enabled:
        internal_key = get_or_create_internal_key()
        return {"api_key": internal_key}
    return {"api_key": None}


@app.get("/api/logs")
async def get_logs(lines: int = 100):
    """Get recent container logs"""
    try:
        # Read from the standard output/error logs
        import subprocess
        
        # Try to get logs from the running container
        result = subprocess.run(
            ["tail", "-n", str(lines), "/proc/1/fd/1"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logs = result.stdout
        else:
            # Fallback: try to read from Python's logging
            logs = "Unable to read container logs directly. Check docker-compose logs."
        
        return {
            "logs": logs,
            "lines": lines,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return {
            "logs": f"Error reading logs: {str(e)}",
            "lines": 0,
            "timestamp": datetime.now().isoformat()
        }


@app.post("/api/auth/dashboard")
async def update_dashboard_auth(request: Dict[str, Any]):
    """Update dashboard authentication settings"""
    enabled = request.get("enabled", False)
    password = request.get("password")
    
    config = load_auth_config()
    config.dashboard_auth_enabled = enabled
    
    if enabled and password:
        config.dashboard_password_hash = hash_password(password)
    elif not enabled:
        config.dashboard_password_hash = None
    
    if save_auth_config(config):
        return {
            "status": "success",
            "message": f"Dashboard authentication {'enabled' if enabled else 'disabled'}"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to save authentication settings")


@app.post("/api/auth/api")
async def update_api_auth(request: Dict[str, bool]):
    """Update API authentication settings"""
    enabled = request.get("enabled", False)
    
    config = load_auth_config()
    config.api_auth_enabled = enabled
    
    if save_auth_config(config):
        return {
            "status": "success",
            "message": f"API authentication {'enabled' if enabled else 'disabled'}"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to save authentication settings")


@app.post("/api/auth/keys")
async def create_api_key(request: Dict[str, str]):
    """Create a new API key"""
    name = request.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="API key name is required")
    
    api_key = add_api_key(name)
    if api_key:
        return {
            "status": "success",
            "api_key": api_key,
            "name": name
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to create API key")


@app.get("/api/auth/keys")
async def list_api_keys(include_full_keys: bool = True):
    """List all API keys with their stats"""
    # Note: Setting include_full_keys=True by default as user requested
    # This is less secure but provides better usability
    return get_api_key_stats(include_full_keys=include_full_keys)


@app.delete("/api/auth/keys/{name}")
async def delete_api_key(name: str):
    """Delete an API key by name"""
    config = load_auth_config()
    
    # Find the key by name
    key_to_delete = None
    for key, info in config.api_keys.items():
        if info.name == name:
            key_to_delete = key
            break
    
    if key_to_delete and remove_api_key(key_to_delete):
        return {
            "status": "success",
            "message": f"API key '{name}' deleted"
        }
    else:
        raise HTTPException(status_code=404, detail="API key not found")


# Catch-all route for debugging
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path: str):
    """Log any unmatched routes for debugging"""
    body = None
    try:
        body = await request.body()
        body_str = body.decode() if body else "No body"
    except:
        body_str = "Could not read body"
    
    logger.warning(f"Unmatched route - Method: {request.method}, Path: /{path}, Headers: {dict(request.headers)}, Body: {body_str[:200]}")
    raise HTTPException(status_code=404, detail="Not Found")


if __name__ == "__main__":
    # Show startup configuration info
    from startup_info import print_startup_info
    print_startup_info()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    )
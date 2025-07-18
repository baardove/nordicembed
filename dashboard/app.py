import os
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI(title="Norwegian Embeddings Dashboard")

templates = Jinja2Templates(directory="dashboard/templates")
app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

API_URL = os.getenv("API_URL", "http://localhost:6000")


async def fetch_api_status() -> Dict[str, Any]:
    """Fetch status from the embedding API."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{API_URL}/") as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "online", **data}
    except Exception as e:
        return {"status": "offline", "error": str(e)}


async def test_embedding(text: str) -> Dict[str, Any]:
    """Test embedding generation."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_URL}/embed",
                json={"texts": [text]}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "embedding_size": len(data["embeddings"][0]),
                        "sample": data["embeddings"][0][:5]  # First 5 dimensions
                    }
                else:
                    return {"success": False, "error": await response.text()}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    api_status = await fetch_api_status()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "api_status": api_status}
    )


@app.post("/test-embedding")
async def test_embedding_endpoint(text: str = Form(...)):
    """Test endpoint for embedding generation."""
    result = await test_embedding(text)
    return JSONResponse(content=result)


@app.get("/api/status")
async def api_status():
    """Get API status."""
    status = await fetch_api_status()
    return JSONResponse(content=status)


@app.get("/api/models")
async def get_models():
    """Get available models."""
    models = [
        {"id": "norbert2", "name": "NorBERT-2", "description": "Norwegian BERT for historical/modern text"},
        {"id": "simcse-nb-bert-large", "name": "SimCSE-NB-BERT-large", "description": "Contrastive sentence embeddings"},
        {"id": "norbert3-base", "name": "NorBERT-3 base", "description": "Updated lighter model"}
    ]
    return JSONResponse(content={"models": models})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
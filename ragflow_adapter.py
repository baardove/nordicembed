"""
RAGFlow Adapter for NoEmbed Service

This module provides compatibility between RAGFlow and NoEmbed's reranking models.
RAGFlow expects reranking to work differently than our standard OpenAI-compatible API.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RAGFlowRerankRequest(BaseModel):
    """RAGFlow-style reranking request format"""
    query: str
    docs: List[str] = Field(..., description="List of documents to rerank")


class RAGFlowRerankResponse(BaseModel):
    """RAGFlow-style reranking response format"""
    scores: List[float] = Field(..., description="Relevance scores for each document")


def convert_ragflow_to_standard_rerank(ragflow_request: RAGFlowRerankRequest) -> Dict[str, Any]:
    """Convert RAGFlow reranking request to our standard format"""
    
    # Extract documents from either passages or documents field
    if ragflow_request.passages:
        documents = [p.get("text", p.get("content", "")) for p in ragflow_request.passages]
    else:
        documents = ragflow_request.documents or []
    
    return {
        "model": ragflow_request.model,
        "query": ragflow_request.query,
        "documents": documents,
        "top_n": ragflow_request.top_k,
        "return_documents": ragflow_request.return_documents
    }


def convert_standard_to_ragflow_response(
    standard_response: Dict[str, Any], 
    original_passages: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Convert our standard reranking response to RAGFlow format"""
    
    results = []
    for item in standard_response.get("data", []):
        result = {
            "index": item.get("original_index", item.get("index")),
            "relevance_score": item.get("relevance_score"),
        }
        
        # If we have original passages, include metadata
        if original_passages and item.get("original_index") < len(original_passages):
            original = original_passages[item["original_index"]]
            result["metadata"] = original.get("metadata", {})
            result["text"] = item.get("document", original.get("text", ""))
        else:
            result["text"] = item.get("document", "")
        
        results.append(result)
    
    return {
        "results": results,
        "model": standard_response.get("model"),
        "usage": standard_response.get("usage", {})
    }


# Configuration for RAGFlow compatibility mode
RAGFLOW_COMPAT_CONFIG = {
    # Map RAGFlow model names to our model names if different
    "model_mapping": {
        # Add any model name mappings here if needed
        # "ragflow-name": "our-name"
    },
    
    # Default settings for RAGFlow
    "defaults": {
        "top_k": 10,
        "return_documents": True
    }
}


def is_ragflow_request(headers: Dict[str, str], body: Dict[str, Any]) -> bool:
    """Detect if this is a RAGFlow request based on headers or body structure"""
    
    # Check for RAGFlow-specific headers
    user_agent = headers.get("user-agent", "").lower()
    if "ragflow" in user_agent:
        return True
    
    # Check for RAGFlow-specific request structure
    if "passages" in body and "query" in body:
        return True
    
    # Check if requesting a reranking model through embeddings endpoint
    model = body.get("model", "")
    if model in ["mmarco-minilm-l12", "ms-marco-minilm-l6", "ms-marco-minilm-l12", 
                 "jina-reranker-multilingual", "nordic-reranker"]:
        # This is likely a RAGFlow request for reranking
        return True
    
    return False
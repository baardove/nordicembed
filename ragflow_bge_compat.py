"""
BGE Reranker Compatibility Layer for RAGFlow

This module provides endpoints that mimic BGE reranker API format
to make NoEmbed compatible with RAGFlow's expectations.
"""

import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BGECompatRequest(BaseModel):
    """BGE-style reranking request format expected by RAGFlow"""
    model: str = Field(default="bge-reranker-base", description="Model name")
    query: str = Field(..., description="The search query")
    passages: List[str] = Field(..., description="List of passages to rerank")
    use_fp16: bool = Field(default=False, description="Use FP16 precision")
    normalize: bool = Field(default=True, description="Normalize scores")


class BGECompatResponse(BaseModel):
    """BGE-style reranking response format expected by RAGFlow"""
    scores: List[float] = Field(..., description="Relevance scores for each passage")
    
    
# Model name mappings from BGE names to NoEmbed names
BGE_TO_NOEMBED_MAPPING = {
    "bge-reranker-base": "mmarco-minilm-l12",
    "bge-reranker-large": "ms-marco-minilm-l12",
    "bge-reranker-v2-m3": "jina-reranker-multilingual",
    # Allow direct model names too
    "mmarco-minilm-l12": "mmarco-minilm-l12",
    "ms-marco-minilm-l6": "ms-marco-minilm-l6",
    "ms-marco-minilm-l12": "ms-marco-minilm-l12",
    "jina-reranker-multilingual": "jina-reranker-multilingual",
    "nordic-reranker": "mmarco-minilm-l12"
}


def map_bge_to_noembed_model(bge_model: str) -> str:
    """Map BGE model names to NoEmbed model names"""
    return BGE_TO_NOEMBED_MAPPING.get(bge_model, "mmarco-minilm-l12")
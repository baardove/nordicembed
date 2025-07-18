import os
import logging
from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)


class RerankerService:
    RERANKER_CONFIGS = {
        # Multilingual rerankers
        "mmarco-minilm-l12": {
            "model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "local_path": "mmarco-minilm-l12",
            "description": "Multilingual reranker supporting 14+ languages"
        },
        "ms-marco-minilm-l6": {
            "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "local_path": "ms-marco-minilm-l6",
            "description": "Fast English reranker, works cross-lingually"
        },
        "ms-marco-minilm-l12": {
            "model_name": "cross-encoder/ms-marco-MiniLM-L12-v2",
            "local_path": "ms-marco-minilm-l12",
            "description": "High quality English reranker"
        },
        "jina-reranker-multilingual": {
            "model_name": "jinaai/jina-reranker-v2-base-multilingual",
            "local_path": "jina-reranker-multilingual",
            "description": "Modern multilingual reranker"
        },
        # Language-specific rerankers (if available in future)
        "nordic-reranker": {
            "model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # Using multilingual for now
            "local_path": "nordic-reranker",
            "description": "Nordic languages reranker (using multilingual model)"
        }
    }
    
    def __init__(self, model_name: str, model_path: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        
        if model_name not in self.RERANKER_CONFIGS:
            raise ValueError(f"Unknown reranker model: {model_name}. Available models: {list(self.RERANKER_CONFIGS.keys())}")
        
        self.config = self.RERANKER_CONFIGS[model_name]
        self._load_model()
    
    def _load_model(self):
        local_model_path = os.path.join(self.model_path, self.config["local_path"])
        
        try:
            if os.path.exists(local_model_path):
                logger.info(f"Loading reranker from local path: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            else:
                logger.info(f"Loading reranker from HuggingFace: {self.config['model_name']}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
                self.model = AutoModelForSequenceClassification.from_pretrained(self.config["model_name"])
                
                logger.info(f"Saving reranker to: {local_model_path}")
                os.makedirs(local_model_path, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
        except Exception as e:
            logger.error(f"Failed to load reranker model: {str(e)}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Reranker loaded successfully on {self.device}")
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float, str]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Return only top K documents (None = return all)
            
        Returns:
            List of tuples (original_index, score, document_text) sorted by score descending
        """
        if not documents:
            return []
        
        # Prepare pairs for the cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get scores
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Handle both single logit and multi-class outputs
            if outputs.logits.shape[-1] == 1:
                scores = outputs.logits.squeeze(-1).cpu().numpy()
            else:
                # For multi-class, use the positive class probability
                scores = torch.softmax(outputs.logits, dim=-1)[:, -1].cpu().numpy()
        
        # Create result tuples with original indices
        results = [(i, float(score), doc) for i, (score, doc) in enumerate(zip(scores, documents))]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return results
    
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs.
        
        Args:
            pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        if not pairs:
            return []
        
        # Tokenize
        encoded = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get scores
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Handle both single logit and multi-class outputs
            if outputs.logits.shape[-1] == 1:
                scores = outputs.logits.squeeze(-1).cpu().numpy()
            else:
                # For multi-class, use the positive class probability
                scores = torch.softmax(outputs.logits, dim=-1)[:, -1].cpu().numpy()
        
        return scores.tolist()
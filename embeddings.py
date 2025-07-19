import os
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    MODEL_CONFIGS = {
        "norbert2": {
            "model_name": "ltg/norbert2",
            "local_path": "norbert2",
            "pooling": "mean"
        },
        "nb-bert-base": {
            "model_name": "NbAiLab/nb-bert-base",
            "local_path": "nb-bert-base",
            "pooling": "mean"
        },
        "nb-bert-large": {
            "model_name": "NbAiLab/nb-bert-large",
            "local_path": "nb-bert-large",
            "pooling": "mean"
        },
        "simcse-nb-bert-large": {
            "model_name": "FFI/SimCSE-NB-BERT-large",
            "local_path": "simcse-nb-bert-large",
            "pooling": "cls"
        },
        "norbert3-base": {
            "model_name": "ltg/norbert3-base",
            "local_path": "norbert3-base",
            "pooling": "mean",
            "trust_remote_code": True
        },
        "norbert3-large": {
            "model_name": "ltg/norbert3-large",
            "local_path": "norbert3-large",
            "pooling": "mean",
            "trust_remote_code": True
        },
        "xlm-roberta-base": {
            "model_name": "xlm-roberta-base",
            "local_path": "xlm-roberta-base",
            "pooling": "mean"
        },
        "electra-small-nordic": {
            "model_name": "ltg/electra-small-nordic",
            "local_path": "electra-small-nordic",
            "pooling": "mean"
        },
        "sentence-bert-base": {
            "model_name": "NbAiLab/sentence-bert-base",
            "local_path": "sentence-bert-base",
            "pooling": "mean"
        },
        # Swedish models
        "kb-sbert-swedish": {
            "model_name": "KBLab/sentence-bert-swedish-cased",
            "local_path": "kb-sbert-swedish",
            "pooling": "mean"
        },
        "kb-bert-swedish": {
            "model_name": "KB/bert-base-swedish-cased",
            "local_path": "kb-bert-swedish",
            "pooling": "mean"
        },
        "bert-large-swedish": {
            "model_name": "AI-Nordics/bert-large-swedish-cased",
            "local_path": "bert-large-swedish",
            "pooling": "mean"
        },
        "albert-swedish": {
            "model_name": "KBLab/albert-base-swedish-cased-alpha",
            "local_path": "albert-swedish",
            "pooling": "mean"
        },
        # Danish models
        "dabert": {
            "model_name": "Maltehb/danish-bert-botxo",
            "local_path": "dabert",
            "pooling": "mean"
        },
        "aelaectra-danish": {
            "model_name": "Maltehb/aelaectra-danish-electra-small-cased",
            "local_path": "aelaectra-danish",
            "pooling": "mean"
        },
        "da-bert-ner": {
            "model_name": "DaNLP/da-bert-ner",
            "local_path": "da-bert-ner",
            "pooling": "mean"
        },
        "electra-base-danish": {
            "model_name": "Maltehb/electra-base-danish-cased",
            "local_path": "electra-base-danish",
            "pooling": "mean"
        },
        # Multilingual Nordic models
        "multilingual-e5-base": {
            "model_name": "intfloat/multilingual-e5-base",
            "local_path": "multilingual-e5-base",
            "pooling": "mean"
        },
        "paraphrase-multilingual-minilm": {
            "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "local_path": "paraphrase-multilingual-minilm",
            "pooling": "mean"
        },
        # Finnish models
        "finbert-base": {
            "model_name": "TurkuNLP/bert-base-finnish-cased-v1",
            "local_path": "finbert-base",
            "pooling": "mean"
        },
        "finbert-sbert": {
            "model_name": "TurkuNLP/sbert-cased-finnish-paraphrase",
            "local_path": "finbert-sbert",
            "pooling": "mean"
        },
        "finbert-large": {
            "model_name": "TurkuNLP/bert-large-finnish-cased-v1",
            "local_path": "finbert-large",
            "pooling": "mean"
        },
        # Icelandic model
        "icebert": {
            "model_name": "mideind/IceBERT",
            "local_path": "icebert",
            "pooling": "mean"
        }
    }
    
    def __init__(self, model_name: str, model_path: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[model_name]
        self._load_model()
    
    def _load_model(self):
        local_model_path = os.path.join(self.model_path, self.config["local_path"])
        
        # Check if model requires trust_remote_code
        trust_remote_code = self.config.get("trust_remote_code", False)
        
        # Check global setting
        from config import get_settings
        settings = get_settings()
        if trust_remote_code and not settings.allow_trust_remote_code:
            raise ValueError(f"Model {self.model_name} requires trust_remote_code=True, but ALLOW_TRUST_REMOTE_CODE is disabled. Set ALLOW_TRUST_REMOTE_CODE=true in your .env file to enable.")
        
        try:
            if os.path.exists(local_model_path):
                logger.info(f"Loading model from local path: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
                self.model = AutoModel.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
            else:
                logger.info(f"Loading model from HuggingFace: {self.config['model_name']}")
                if trust_remote_code:
                    logger.warning(f"Model {self.config['model_name']} requires trust_remote_code=True. This will execute custom code from the model repository.")
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"], trust_remote_code=trust_remote_code)
                self.model = AutoModel.from_pretrained(self.config["model_name"], trust_remote_code=trust_remote_code)
                
                logger.info(f"Saving model to: {local_model_path}")
                os.makedirs(local_model_path, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _cls_pooling(self, model_output):
        return model_output[0][:, 0]
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        if self.config["pooling"] == "mean":
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        elif self.config["pooling"] == "cls":
            embeddings = self._cls_pooling(model_output)
        else:
            raise ValueError(f"Unknown pooling method: {self.config['pooling']}")
        
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().tolist()
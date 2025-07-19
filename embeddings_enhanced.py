import os
import logging
import time
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError

logger = logging.getLogger(__name__)


def diagnose_connection_error(model_name: str, error: Exception) -> str:
    """Provide detailed diagnostics for connection errors."""
    error_msg = str(error)
    
    diagnostics = [f"Failed to load model '{model_name}'"]
    
    if isinstance(error, ConnectionError):
        diagnostics.append("Connection Error Details:")
        diagnostics.append("- Unable to connect to HuggingFace servers")
        diagnostics.append("- Check your internet connection")
        diagnostics.append("- Check if you're behind a proxy/firewall")
        diagnostics.append("- Try: export HF_HUB_OFFLINE=1 if models are already downloaded")
    
    elif isinstance(error, Timeout):
        diagnostics.append("Timeout Error Details:")
        diagnostics.append("- Download took too long")
        diagnostics.append("- This model is large and may need more time")
        diagnostics.append("- Try increasing timeout or downloading manually")
    
    elif isinstance(error, HTTPError):
        diagnostics.append(f"HTTP Error: {error_msg}")
        if "404" in error_msg:
            diagnostics.append("- Model not found on HuggingFace")
            diagnostics.append("- Check model name spelling")
        elif "403" in error_msg:
            diagnostics.append("- Access forbidden")
            diagnostics.append("- Model might require authentication")
        elif "503" in error_msg:
            diagnostics.append("- HuggingFace service temporarily unavailable")
            diagnostics.append("- Try again later")
    
    elif "trust_remote_code" in error_msg:
        diagnostics.append("Trust Remote Code Error:")
        diagnostics.append("- This model requires custom code execution")
        diagnostics.append("- Ensure ALLOW_TRUST_REMOTE_CODE=true in .env")
        diagnostics.append("- Restart the service after changing settings")
    
    elif "CUDA" in error_msg or "GPU" in error_msg:
        diagnostics.append("GPU/CUDA Error:")
        diagnostics.append("- Model requires GPU but none available")
        diagnostics.append("- Switch to CPU mode in settings")
    
    elif "disk space" in error_msg.lower() or "no space" in error_msg.lower():
        diagnostics.append("Storage Error:")
        diagnostics.append("- Insufficient disk space")
        diagnostics.append("- Models can be 1-5GB each")
        diagnostics.append("- Clear space or change MODEL_PATH directory")
    
    else:
        diagnostics.append(f"Error details: {error_msg}")
        diagnostics.append("\nCommon solutions:")
        diagnostics.append("- Check internet connectivity")
        diagnostics.append("- Verify model name is correct")
        diagnostics.append("- Ensure sufficient disk space")
        diagnostics.append("- Check Docker logs: docker-compose logs embed-nordic")
    
    return "\n".join(diagnostics)


def download_with_retry(download_func, max_retries=3, delay=5):
    """Retry download function with exponential backoff."""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Download attempt {attempt + 1}/{max_retries}")
            return download_func()
        except (ConnectionError, Timeout, HTTPError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Download failed, retrying in {wait_time}s... Error: {str(e)}")
                time.sleep(wait_time)
            else:
                logger.error(f"Download failed after {max_retries} attempts")
                raise
    
    raise last_error


class EmbeddingServiceEnhanced:
    """Enhanced embedding service with better error handling."""
    
    MODEL_CONFIGS = {
        # Copy from original embeddings.py
        "norbert3-large": {
            "model_name": "ltg/norbert3-large",
            "local_path": "norbert3-large",
            "pooling": "mean",
            "trust_remote_code": True
        },
        # ... other models ...
    }
    
    def __init__(self, model_name: str, model_path: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODEL_CONFIGS.keys())}")
        
        self.config = self.MODEL_CONFIGS[model_name]
        self._load_model_with_diagnostics()
    
    def _load_model_with_diagnostics(self):
        """Load model with enhanced error diagnostics."""
        local_model_path = os.path.join(self.model_path, self.config["local_path"])
        
        # Check if model requires trust_remote_code
        trust_remote_code = self.config.get("trust_remote_code", False)
        
        # Check global setting
        from config import get_settings
        settings = get_settings()
        if trust_remote_code and not settings.allow_trust_remote_code:
            error_msg = (
                f"Model {self.model_name} requires trust_remote_code=True, "
                f"but ALLOW_TRUST_REMOTE_CODE is disabled.\n"
                f"Solution: Set ALLOW_TRUST_REMOTE_CODE=true in your .env file and restart."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            if os.path.exists(local_model_path):
                logger.info(f"Loading model from local path: {local_model_path}")
                self._load_from_local(local_model_path, trust_remote_code)
            else:
                logger.info(f"Model not found locally, downloading from HuggingFace: {self.config['model_name']}")
                if trust_remote_code:
                    logger.warning(
                        f"Model {self.config['model_name']} requires trust_remote_code=True. "
                        f"This will execute custom code from the model repository."
                    )
                
                # Download with retry mechanism
                def download_model():
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config["model_name"], 
                        trust_remote_code=trust_remote_code
                    )
                    self.model = AutoModel.from_pretrained(
                        self.config["model_name"], 
                        trust_remote_code=trust_remote_code
                    )
                    return True
                
                download_with_retry(download_model)
                
                # Save model after successful download
                logger.info(f"Saving model to: {local_model_path}")
                os.makedirs(local_model_path, exist_ok=True)
                self.tokenizer.save_pretrained(local_model_path)
                self.model.save_pretrained(local_model_path)
                
        except Exception as e:
            error_diagnostics = diagnose_connection_error(self.model_name, e)
            logger.error(error_diagnostics)
            raise RuntimeError(error_diagnostics) from e
        
        # Move model to device
        try:
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                logger.error(f"Failed to load model on {self.device}: {str(e)}")
                logger.error("Try setting DEVICE=cpu in your .env file")
            raise
    
    def _load_from_local(self, local_model_path: str, trust_remote_code: bool):
        """Load model from local path."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_path, 
            trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            local_model_path, 
            trust_remote_code=trust_remote_code
        )
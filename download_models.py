#!/usr/bin/env python3
"""
Download Norwegian embedding models for local use.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS = {
    "norbert2": {
        "hf_name": "ltg/norbert2",
        "description": "Norwegian BERT; good for historical/modern text"
    },
    "nb-bert-base": {
        "hf_name": "NbAiLab/nb-bert-base",
        "description": "Norwegian BERT base model by National Library of Norway"
    },
    "nb-bert-large": {
        "hf_name": "NbAiLab/nb-bert-large",
        "description": "Large Norwegian BERT model for higher quality embeddings"
    },
    "simcse-nb-bert-large": {
        "hf_name": "FFI/SimCSE-NB-BERT-large",
        "description": "Contrastive sentence embeddings in Norwegian"
    },
    "norbert3-base": {
        "hf_name": "ltg/norbert3-base",
        "description": "Updated lighter Norwegian BERT model"
    },
    "norbert3-large": {
        "hf_name": "ltg/norbert3-large",
        "description": "Latest NorBERT iteration, large size for best quality"
    },
    "xlm-roberta-base": {
        "hf_name": "xlm-roberta-base",
        "description": "Multilingual model including Norwegian"
    },
    "electra-small-nordic": {
        "hf_name": "ltg/electra-small-nordic",
        "description": "Small ELECTRA model for fast Nordic language inference"
    },
    "sentence-bert-base": {
        "hf_name": "NbAiLab/sentence-bert-base",
        "description": "Sentence embeddings optimized for similarity tasks"
    },
    # Swedish models
    "kb-sbert-swedish": {
        "hf_name": "KBLab/sentence-bert-swedish-cased",
        "description": "Swedish sentence embeddings optimized for similarity"
    },
    "kb-bert-swedish": {
        "hf_name": "KB/bert-base-swedish-cased",
        "description": "Swedish BERT by National Library of Sweden"
    },
    "bert-large-swedish": {
        "hf_name": "AI-Nordics/bert-large-swedish-cased",
        "description": "Large Swedish BERT model for high quality embeddings"
    },
    "albert-swedish": {
        "hf_name": "KBLab/albert-base-swedish-cased-alpha",
        "description": "Swedish ALBERT - smaller and faster than BERT"
    },
    # Danish models
    "dabert": {
        "hf_name": "Maltehb/danish-bert-botxo",
        "description": "Danish BERT (DaBERT) model"
    },
    "aelaectra-danish": {
        "hf_name": "Maltehb/aelaectra-danish-electra-small-cased",
        "description": "Small, efficient Danish ELECTRA model"
    },
    "da-bert-ner": {
        "hf_name": "DaNLP/da-bert-ner",
        "description": "Danish BERT fine-tuned for NER, also good for general embeddings"
    },
    "electra-base-danish": {
        "hf_name": "Maltehb/electra-base-danish-cased",
        "description": "Efficient Danish ELECTRA base model"
    },
    # Multilingual Nordic models
    "multilingual-e5-base": {
        "hf_name": "intfloat/multilingual-e5-base",
        "description": "Multilingual E5 embeddings supporting Nordic languages"
    },
    "paraphrase-multilingual-minilm": {
        "hf_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "description": "Multilingual sentence embeddings (50+ languages)"
    },
    # Finnish models
    "finbert-base": {
        "hf_name": "TurkuNLP/bert-base-finnish-cased-v1",
        "description": "Finnish BERT base model by TurkuNLP"
    },
    "finbert-sbert": {
        "hf_name": "TurkuNLP/sbert-cased-finnish-paraphrase",
        "description": "Finnish sentence embeddings for similarity"
    },
    "finbert-large": {
        "hf_name": "TurkuNLP/bert-large-finnish-cased-v1",
        "description": "Large Finnish BERT for high quality"
    },
    # Icelandic model
    "icebert": {
        "hf_name": "mideind/IceBERT",
        "description": "Icelandic RoBERTa-base model"
    }
}


def download_model(model_name: str, base_path: str = "./models"):
    """Download a specific model from HuggingFace."""
    if model_name not in MODELS:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_name]
    model_path = Path(base_path) / model_name
    
    if model_path.exists() and any(model_path.iterdir()):
        logger.info(f"Model {model_name} already exists at {model_path}")
        response = input("Do you want to re-download? (y/N): ")
        if response.lower() != 'y':
            return True
    
    logger.info(f"Downloading {model_name}: {model_info['description']}")
    logger.info(f"From HuggingFace: {model_info['hf_name']}")
    
    try:
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_info['hf_name'])
        tokenizer.save_pretrained(model_path)
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModel.from_pretrained(model_info['hf_name'])
        model.save_pretrained(model_path)
        
        logger.info(f"Successfully downloaded {model_name} to {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Norwegian embedding models")
    parser.add_argument(
        "models",
        nargs="*",
        choices=list(MODELS.keys()) + ["all"],
        help="Models to download (default: all)"
    )
    parser.add_argument(
        "--path",
        default="./models",
        help="Base path for model storage (default: ./models)"
    )
    
    args = parser.parse_args()
    
    if not args.models or "all" in args.models:
        models_to_download = list(MODELS.keys())
    else:
        models_to_download = args.models
    
    logger.info(f"Will download models: {', '.join(models_to_download)}")
    
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, args.path):
            success_count += 1
    
    logger.info(f"Successfully downloaded {success_count}/{len(models_to_download)} models")
    
    if success_count < len(models_to_download):
        sys.exit(1)


if __name__ == "__main__":
    main()
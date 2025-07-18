#!/usr/bin/env python3
"""
Test script to verify all Norwegian embedding models are working correctly.
"""
import os
import sys
import logging
from embeddings import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test texts in Norwegian
TEST_TEXTS = [
    "Norge er et land i Nord-Europa.",
    "Oslo er hovedstaden i Norge.",
    "Fjordene er vakre om sommeren."
]

def test_model(model_name: str, model_path: str = "./models"):
    """Test a single model."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing model: {model_name}")
    logger.info(f"{'='*50}")
    
    try:
        # Initialize the embedding service
        service = EmbeddingService(
            model_name=model_name,
            model_path=model_path,
            device="cpu",  # Use CPU for testing
            max_length=512
        )
        logger.info(f"✓ Model loaded successfully")
        
        # Test embedding generation
        embeddings = service.embed(TEST_TEXTS)
        logger.info(f"✓ Generated embeddings for {len(TEST_TEXTS)} texts")
        
        # Check embedding dimensions
        embedding_dim = len(embeddings[0])
        logger.info(f"✓ Embedding dimension: {embedding_dim}")
        
        # Verify all embeddings have the same dimension
        for i, emb in enumerate(embeddings):
            if len(emb) != embedding_dim:
                raise ValueError(f"Embedding {i} has different dimension: {len(emb)} != {embedding_dim}")
        
        logger.info(f"✓ All embeddings have consistent dimensions")
        
        # Calculate and display first few values
        logger.info(f"✓ Sample embedding values (first 5): {embeddings[0][:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to test model {model_name}: {str(e)}")
        return False


def main():
    """Test all available models."""
    models_to_test = [
        "norbert2",
        "nb-bert-base",
        "nb-bert-large",
        "simcse-nb-bert-large",
        "norbert3-base",
        "norbert3-large",
        "xlm-roberta-base",
        "electra-small-nordic",
        "sentence-bert-base"
    ]
    
    logger.info(f"Testing {len(models_to_test)} Norwegian embedding models...")
    logger.info(f"Note: Models will be auto-downloaded if not present locally")
    
    results = {}
    for model_name in models_to_test:
        success = test_model(model_name)
        results[model_name] = success
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    
    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful
    
    for model_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{model_name:.<30} {status}")
    
    logger.info(f"\nTotal: {successful}/{len(results)} models passed")
    
    if failed > 0:
        logger.error(f"\n{failed} models failed testing")
        sys.exit(1)
    else:
        logger.info("\nAll models passed testing!")


if __name__ == "__main__":
    main()
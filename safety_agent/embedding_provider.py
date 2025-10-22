
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from gcp_config import gcp_config
import os

def get_embedding_model(model_provider_name: str):
    if model_provider_name == 'vertex_ai':
        model = vertex_ai_embed_model()
        if model is None:
            print("⚠️ Vertex AI embedding model not available")
            return None
        return model
    else:
        print(f"⚠️ Invalid embedding model provider: {model_provider_name}")
        return None

def vertex_ai_embed_model(model: str | None = None):
    """Initialize Vertex AI text embedding model.

    The function will use the explicit `model` argument if provided, otherwise
    it will read `EMBEDDING_MODEL_NAME` from the environment. On failure it
    returns None so callers can gracefully fall back.
    """
    try:
        # Allow overriding the model via environment variable
        env_model = os.getenv("EMBEDDING_MODEL_NAME")
        model_name = model or env_model or "text-embedding-004"

        # Get GCP configuration
        config = gcp_config.get_vertex_ai_config()

        # Explicitly set the project ID and location for Vertex AI
        aiplatform.init(
            project=config['project_id'],
            location=config['location']
        )

        # Initialize the embedding model
        embedding_model = TextEmbeddingModel.from_pretrained(model_name)
        return embedding_model

    except Exception as e:
        # Helpful debug message for common model-not-found issues
        print(f"⚠️ Error initializing Vertex AI embedding model '{model_name}': {e}")
        print("Please ensure the model name is correct and your project has access to it.")
        print("You can list available models in your project/region with the gcloud CLI or check the Vertex AI Models page:")
        print("  - Console: https://console.cloud.google.com/vertex-ai/models?project=<PROJECT_ID>")
        print("  - gcloud (example for us-central1):")
        print("      gcloud ai models list --region=us-central1 --project=YOUR_PROJECT_ID")
        # Return None instead of raising to allow graceful degradation
        return None

def get_embeddings(texts, model="textembedding-gecko@003"):
    """Get embeddings for a list of texts"""
    try:
        embedding_model = vertex_ai_embed_model(model)
        
        # Check if embedding model is available
        if embedding_model is None:
            print("⚠️ Embedding model not available")
            return None
        
        # Get embeddings
        embeddings = embedding_model.get_embeddings(texts)
        
        # Extract the embedding values
        embedding_values = [embedding.values for embedding in embeddings]
        
        return embedding_values
        
    except Exception as e:
        print(f"⚠️ Error getting embeddings: {e}")
        return None

import os
from google.cloud import aiplatform
from google.cloud import firestore
from google.cloud import storage
from google.cloud import documentai
from google.auth import default
from dotenv import load_dotenv

# Load production environment variables

load_dotenv()

class GCPConfig:
    """GCP Configuration and initialization class"""
    
    def __init__(self):
        self.project_id = os.getenv("SAFETY_GCP_PROJECT_ID", "safety-agent-469708")
        self.location = os.getenv("GCP_LOCATION", "us-central1")
        self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "safety-agent-469708-651aa87c4a1a.json")

        print(f"🔧 GCP Config - Project ID: {self.project_id}")
        print(f"🔧 GCP Config - Location: {self.location}")
        print(f"🔧 GCP Config - Credentials: {self.credentials_path}")

    # Initialize GCP clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize GCP service clients"""
        try:
            # Set credentials
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.project_id,
                location=self.location,
            )
            
            # Initialize Firestore
            self.firestore_client = firestore.Client(project=self.project_id)
            
            # Initialize Cloud Storage
            self.storage_client = storage.Client(project=self.project_id)
            
            # Initialize Document AI
            self.documentai_client = documentai.DocumentProcessorServiceClient()
            
            print("✅ GCP clients initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing GCP clients: {e}")
            raise
    
    def get_vertex_ai_config(self):
        """Get Vertex AI configuration"""
        return {
            "project_id": self.project_id,
            "location": self.location,
            "model": os.getenv("VERTEX_AI_MODEL", "gemini-2.0-flash"),
            "embedding_model": os.getenv("VERTEX_AI_EMBEDDING_MODEL", "text-embedding-004"),
            "temperature": float(os.getenv("VERTEX_AI_TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("VERTEX_AI_MAX_TOKENS", "1024")),
            "top_k": int(os.getenv("VERTEX_AI_TOP_K", "40")),
            "top_p": float(os.getenv("VERTEX_AI_TOP_P", "0.95"))
        }
    
    def get_firestore_config(self):
        """Get Firestore configuration"""
        return {
            "project_id": self.project_id,
            "database": os.getenv("FIRESTORE_DATABASE", "(default)"),
            # general collection name (kept for backwards compatibility)
            "collection": os.getenv("FIRESTORE_COLLECTION", "investment_documents"),
            # specific collections used by chat history
            "user_collection": os.getenv("FIRESTORE_USER_COLLECTION", "users"),
            "chat_collection": os.getenv("FIRESTORE_CHAT_COLLECTION", "chats")
        }
    
    def get_storage_config(self):
        """Get Cloud Storage configuration"""
        return {
            "project_id": self.project_id,
            "bucket_name": os.getenv("GCS_BUCKET_NAME", "gs://investment-agent-documents"),
            "temp_folder": os.getenv("GCS_TEMP_FOLDER", "tmp"),
            "max_retries": int(os.getenv("GCS_MAX_RETRIES", "3"))
        }
    
    def get_vector_search_config(self):
        """Get Vector Search configuration"""
        return {
            "vector_index": os.getenv("VECTOR_INDEX", "1000979992764481536"),
            "multimodal_index": os.getenv("MULTIMODAL_INDEX", "3817981559684726784"),
            "location": self.location
        }
    
    def get_document_ai_config(self):
        """Get Document AI configuration"""
        return {
            "project_id": self.project_id,
            "location": os.getenv("DOCUMENT_AI_LOCATION", "us"),
            "processor_id": os.getenv("DOCUMENT_AI_PROCESSOR_ID", "")
        }

# Global GCP config instance
gcp_config = GCPConfig() 
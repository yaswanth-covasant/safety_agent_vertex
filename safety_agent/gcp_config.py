import os
import json
from google.cloud import aiplatform, firestore, storage
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load production environment variables
load_dotenv()

class GCPConfig:
    """GCP Configuration and initialization class"""
    
    def __init__(self):
        # The project ID is fundamental and should be loaded first.
        self.project_id = os.getenv("SAFETY_GCP_PROJECT_ID", "safety-agent-469708")
        self.location = os.getenv("GCP_LOCATION", "us-central1")

        print(f"🔧 GCP Config - Project ID: {self.project_id}")
        print(f"🔧 GCP Config - Location: {self.location}")
        
        # This will generate credentials and initialize all clients.
        self._init_clients()

    def _get_credentials(self) -> service_account.Credentials:
        """
        Constructs GCP credentials from environment variables.
        This is a private helper method.
        """
        # Ensure private_key is correctly formatted (it often gets mangled by env vars)
        private_key = os.getenv("private_key")

        service_account_info = {
            "type": "service_account", # This field is required by from_service_account_info
            "project_id": self.project_id, # Use the project_id from the class
            "private_key_id": os.getenv("private_key_id"),
            "private_key": os.getenv("private_key"),
            "client_email": os.getenv("client_email"),
            "client_id": os.getenv("client_id"),
            "auth_uri": os.getenv("auth_uri"),
            "token_uri": os.getenv("token_uri"),
            "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
            "client_x509_cert_url": os.getenv("client_x509_cert_url"),
            "universe_domain": os.getenv("universe_domain")
        }
        
      
        print("🔧 GCP Config - Successfully loaded credentials from environment variables.")
        return service_account.Credentials.from_service_account_info(service_account_info)
    
    def _init_clients(self):
        """Initialize GCP service clients using credentials from environment variables."""
        try:
            # Step 1: Generate the credentials object
            credentials = self._get_credentials()
            print("-------cred------")
            print(credentials)
            
            # Step 2: Initialize all clients using the same credentials object
            
            # Initialize Vertex AI
            aiplatform.init(
                project=self.project_id,
                location=self.location,
                credentials=credentials,
            )
            
            # Initialize Firestore
            self.firestore_client = firestore.Client(
                project=self.project_id, 
                credentials=credentials
            )
            
            # Initialize Cloud Storage
            self.storage_client = storage.Client(
                project=self.project_id, 
                credentials=credentials
            )
            
            # Initialize Document AI
            # The client_options are needed if the Document AI endpoint is not in the default location.
            # docai_location = os.getenv("DOCUMENT_AI_LOCATION", "us")
            # client_options = {"api_endpoint": f"{docai_location}-documentai.googleapis.com"}
            # self.documentai_client = documentai.DocumentProcessorServiceClient(
            #     credentials=credentials,
            #     client_options=client_options
            # )
            
            print("✅ GCP clients initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing GCP clients: {e}")
            raise
    
    # ... The rest of your get_..._config methods remain unchanged ...
    
    def get_vertex_ai_config(self):
        """Get Vertex AI configuration"""
        return {
            "project_id": self.project_id,
            "location": self.location,
            "model": os.getenv("VERTEX_AI_MODEL", "gemini-1.5-flash"),
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
            "collection": os.getenv("FIRESTORE_COLLECTION", "investment_documents"),
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
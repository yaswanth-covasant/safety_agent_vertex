

#script of embedding_provider.py


from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
# from gcp_config import gcp_config
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
        config = get_gcp_config().get_vertex_ai_config()

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





#script for gcp_config.py

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


def get_gcp_config():
    """Get or create GCP config instance with lazy initialization"""
    return GCPConfig()

#script of vectordb_provider.py

from google.cloud import firestore
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
# from gcp_config import gcp_config
# from bigquery_client import BigQueryClient
import numpy as np
from typing import List, Dict, Any
import uuid
from langchain_core.documents import Document
import time
import os
import json
from datetime import datetime



# Initialize GCP clients

# firestore_client = get_gcp_config().firestore_client
# storage_client = get_gcp_config().storage_client

def get_vector_store_text(vector_store_provider: str, embedding_model):
    """Get text vector store"""
    if embedding_model is None:
        print("⚠️ Embedding model is None, cannot create vector store")
        return None
        
    provider = vector_store_provider.lower()
    
    if provider == "firestore":
        return _get_firestore_vector_store_text(embedding_model)
    elif provider == "vertex_ai":
        return _get_vertex_ai_vector_store_text(embedding_model)
    else:
        raise ValueError(f"Unknown vector store: {vector_store_provider}")

def get_vector_store_multimodal(vector_store_provider: str, embedding_model):
    """Get multimodal vector store"""
    if embedding_model is None:
        print("⚠️ Embedding model is None, cannot create vector store")
        return None
        
    provider = vector_store_provider.lower()
    
    if provider == "firestore":
        return _get_firestore_vector_store_multimodal(embedding_model)
    elif provider == "vertex_ai":
        return _get_vertex_ai_vector_store_multimodal(embedding_model)
    else:
        raise ValueError(f"Unknown vector store: {vector_store_provider}")

def _get_vertex_ai_vector_store_text(embedding_model):
    """Create Vertex AI Vector Search-based text vector store"""
    config = get_gcp_config().get_vector_search_config()
    
    class VertexAIVectorStore:
        def __init__(self, embedding_model, index_id=None):
            self.embedding_model = embedding_model
            self.index_id = index_id or config['index_id']
            self.project_id = get_gcp_config().project_id
            self.location = get_gcp_config().location
            
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Get or create index endpoint
            self.index_endpoint = self._get_or_create_index_endpoint()
        
        def _get_or_create_index_endpoint(self):
            """Get existing index endpoint or create new one"""
            try:
                # Try to get existing endpoint
                index_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
                for endpoint in index_endpoints:
                    if endpoint.display_name == self.index_id:
                        print(f"✅ Using existing Vector Search endpoint: {endpoint.name}")
                        return endpoint
                
                # Create new endpoint if none exists
                print(f"🔧 Creating new Vector Search endpoint: {self.index_id}")
                endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                    display_name=self.index_id,
                    description="Vector search endpoint for RAG application"
                )
                return endpoint
                
            except Exception as e:
                print(f"⚠️ Using Firestore fallback due to Vector Search error: {e}")
                return None
        
        def similarity_search_with_score(self, query: str, k: int = 5):
            """Search for similar documents with scores using Vertex AI Vector Search"""
            try:
                if self.index_endpoint is None:
                    # Fallback to Firestore
                    return _get_firestore_vector_store_text(self.embedding_model).similarity_search_with_score(query, k)
                
                # Get query embedding
                query_embedding = self.embedding_model.get_embeddings([query])[0].values
                
                # Perform vector search
                response = self.index_endpoint.find_neighbors(
                    deployed_index_id=self.index_endpoint.deployed_indexes[0].id,
                    queries=[query_embedding],
                    num_neighbors=k
                )
                
                # Process results
                docs = []
                scores = []
                
                for neighbor in response.nearest_neighbors[0].neighbors:
                    # Get document from Firestore using neighbor id
                    doc_ref = get_gcp_config().firestore_client.collection('documents').document(neighbor.id)
                    doc_data = doc_ref.get()
                    
                    if doc_data.exists:
                        doc_dict = doc_data.to_dict()
                        document = Document(
                            page_content=doc_dict.get('content', ''),
                            metadata={
                                'source': doc_dict.get('source', ''),
                                'page': doc_dict.get('page', 0),
                                'doc_id': neighbor.id,
                                'score': neighbor.distance
                            }
                        )
                        docs.append(document)
                        scores.append(1 - neighbor.distance)  # Convert distance to similarity
                
                # Sort by score
                sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return sorted_pairs[:k]
                
            except Exception as e:
                print(f"Error in Vertex AI vector search: {e}, falling back to Firestore")
                return _get_firestore_vector_store_text(self.embedding_model).similarity_search_with_score(query, k)
        
        def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
            """Add documents with embeddings to both Firestore and Vector Search"""
            try:
                # Add to Firestore first
                firestore_store = _get_firestore_vector_store_text(self.embedding_model)
                firestore_store.add_documents(documents, embeddings)
                
                # TODO: Add to Vector Search index (requires batch processing)
                print("✅ Documents added to Firestore. Vector Search indexing will be handled separately.")
                
            except Exception as e:
                print(f"Error adding documents: {e}")
                raise
    
    return VertexAIVectorStore(embedding_model)

def _get_vertex_ai_vector_store_multimodal(embedding_model):
    """Create Vertex AI Vector Search-based multimodal vector store"""
    config = get_gcp_config().get_vector_search_config()
    
    class VertexAIMultimodalStore:
        def __init__(self, embedding_model, index_id=None):
            self.embedding_model = embedding_model
            self.index_id = index_id or f"{config['index_id']}_multimodal"
            self.project_id = get_gcp_config().project_id
            self.location = get_gcp_config().location
            
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)
            
            # Get or create index endpoint
            self.index_endpoint = self._get_or_create_index_endpoint()
        
        def _get_or_create_index_endpoint(self):
            """Get existing index endpoint or create new one"""
            try:
                # Try to get existing endpoint
                index_endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
                for endpoint in index_endpoints:
                    if endpoint.display_name == self.index_id:
                        print(f"✅ Using existing Multimodal Vector Search endpoint: {endpoint.name}")
                        return endpoint
                
                # Create new endpoint if none exists
                print(f"🔧 Creating new Multimodal Vector Search endpoint: {self.index_id}")
                endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                    display_name=self.index_id,
                    description="Multimodal vector search endpoint for RAG application"
                )
                return endpoint
                
            except Exception as e:
                print(f"⚠️ Using Firestore fallback due to Vector Search error: {e}")
                return None
        
        def similarity_search_with_score(self, query: str, k: int = 5):
            """Search for similar multimodal documents with scores"""
            try:
                if self.index_endpoint is None:
                    # Fallback to Firestore
                    return _get_firestore_vector_store_multimodal(self.embedding_model).similarity_search_with_score(query, k)
                
                # Get query embedding
                query_embedding = self.embedding_model.get_embeddings([query])[0].values
                
                # Perform vector search
                response = self.index_endpoint.find_neighbors(
                    deployed_index_id=self.index_endpoint.deployed_indexes[0].id,
                    queries=[query_embedding],
                    num_neighbors=k
                )
                
                # Process results
                docs = []
                scores = []
                
                for neighbor in response.nearest_neighbors[0].neighbors:
                    # Get document from Firestore using neighbor id
                    doc_ref = get_gcp_config().firestore_client.collection('multimodal_documents').document(neighbor.id)
                    doc_data = doc_ref.get()
                    
                    if doc_data.exists:
                        doc_dict = doc_data.to_dict()
                        document = Document(
                            page_content=doc_dict.get('content', ''),
                            metadata={
                                'source': doc_dict.get('source', ''),
                                'page': doc_dict.get('page', 0),
                                'doc_id': neighbor.id,
                                'image_url': doc_dict.get('image_url', ''),
                                'content_type': doc_dict.get('content_type', 'text'),
                                'score': neighbor.distance
                            }
                        )
                        docs.append(document)
                        scores.append(1 - neighbor.distance)  # Convert distance to similarity
                
                # Sort by score
                sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return sorted_pairs[:k]
                
            except Exception as e:
                print(f"Error in Vertex AI multimodal search: {e}, falling back to Firestore")
                return _get_firestore_vector_store_multimodal(self.embedding_model).similarity_search_with_score(query, k)
        
        def add_multimodal_documents(self, documents: List[Document], embeddings: List[List[float]]):
            """Add multimodal documents with embeddings"""
            try:
                # Add to Firestore first
                firestore_store = _get_firestore_vector_store_multimodal(self.embedding_model)
                firestore_store.add_multimodal_documents(documents, embeddings)
                
                # TODO: Add to Vector Search index (requires batch processing)
                print("✅ Multimodal documents added to Firestore. Vector Search indexing will be handled separately.")
                
            except Exception as e:
                print(f"Error adding multimodal documents: {e}")
                raise
    
    return VertexAIMultimodalStore(embedding_model)

def _get_firestore_vector_store_text(embedding_model):
    """Create Firestore-based text vector store (fallback)"""
    config = get_gcp_config().get_firestore_config()
    
    class FirestoreVectorStore:
        def __init__(self, embedding_model, collection_name="documents"):
            self.embedding_model = embedding_model
            self.collection = get_gcp_config().firestore_client.collection(collection_name)
        
        def similarity_search_with_score(self, query: str, k: int = 5):
            """Search for similar documents with scores"""
            try:
                # Get query embedding
                query_embedding = self.embedding_model.get_embeddings([query])[0].values
                
                # Search in Firestore
                docs = []
                scores = []
                
                # Get all documents and compute similarity
                documents = self.collection.stream()
                
                for doc in documents:
                    doc_data = doc.to_dict()
                    if 'embedding' in doc_data:
                        doc_embedding = np.array(doc_data['embedding'])
                        # Compute cosine similarity
                        similarity = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                        
                        # Create document object
                        document = Document(
                            page_content=doc_data.get('text', ''),
                            metadata={
                                'source': doc_data.get('source', ''),
                                'page': doc_data.get('page', 0),
                                'doc_id': doc.id
                            }
                        )
                        
                        docs.append(document)
                        scores.append(similarity)
                
                # Sort by similarity score and return top k
                sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return sorted_pairs[:k]
                
            except Exception as e:
                print(f"Error in similarity search: {e}")
                return []
        
        def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
            """Add documents with embeddings to Firestore"""
            try:
                for doc, embedding in zip(documents, embeddings):
                    doc_data = {
                        'content': doc.page_content,
                        'embedding': embedding,
                        'source': doc.metadata.get('source', ''),
                        'page': doc.metadata.get('page', 0),
                        'timestamp': firestore.SERVER_TIMESTAMP
                    }
                    
                    self.collection.add(doc_data)
                    
                print(f"✅ Added {len(documents)} documents to Firestore")
                # Optionally mirror documents to BigQuery byte_store (append-only)
                if os.getenv("BIGQUERY_DATASET") and os.getenv("BIGQUERY_BYTESTORE_TABLE"):
                    try:
                        bq = BigQueryClient()
                        rows = []
                        for doc, embedding in zip(documents, embeddings):
                            rows.append({
                                'doc_id': doc.metadata.get('doc_id') or str(uuid.uuid4()),
                                'content': doc.page_content,
                                'embedding': json.dumps(embedding),
                                'metadata': json.dumps(doc.metadata),
                                'created_at': datetime.utcnow().isoformat() + 'Z'
                            })
                        bq.insert_rows(os.getenv("BIGQUERY_BYTESTORE_TABLE"), rows)
                    except Exception as e:
                        print(f"Warning: BigQuery bytestore insert failed: {e}")
                
            except Exception as e:
                print(f"Error adding documents: {e}")
                raise
    
    return FirestoreVectorStore(embedding_model, config['collection'])

def _get_firestore_vector_store_multimodal(embedding_model):
    """Create Firestore-based multimodal vector store (fallback)"""
    config = get_gcp_config().get_firestore_config()
    
    class FirestoreMultimodalStore:
        def __init__(self, embedding_model, collection_name="multimodal_documents"):
            self.embedding_model = embedding_model
            self.collection = get_gcp_config().firestore_client.collection(collection_name)
        
        def similarity_search_with_score(self, query: str, k: int = 5):
            """Search for similar multimodal documents with scores"""
            try:
                # Get query embedding
                query_embedding = self.embedding_model.get_embeddings([query])[0].values
                
                # Search in Firestore
                docs = []
                scores = []
                
                # Get all documents and compute similarity
                documents = self.collection.stream()
                
                for doc in documents:
                    doc_data = doc.to_dict()
                    if 'embedding' in doc_data:
                        doc_embedding = np.array(doc_data['embedding'])
                        # Compute cosine similarity
                        similarity = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                        
                        # Create document object
                        document = Document(
                            page_content=doc_data.get('content', ''),
                            metadata={
                                'source': doc_data.get('source', ''),
                                'page': doc_data.get('page', 0),
                                'doc_id': doc.id,
                                'image_url': doc_data.get('image_url', ''),
                                'content_type': doc_data.get('content_type', 'text')
                            }
                        )
                        
                        docs.append(document)
                        scores.append(similarity)
                
                # Sort by similarity score and return top k
                sorted_pairs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                return sorted_pairs[:k]
                
            except Exception as e:
                print(f"Error in multimodal similarity search: {e}")
                return []
        
        def add_multimodal_documents(self, documents: List[Document], embeddings: List[List[float]]):
            """Add multimodal documents with embeddings to Firestore"""
            try:
                for doc, embedding in zip(documents, embeddings):
                    doc_data = {
                        'content': doc.page_content,
                        'embedding': embedding,
                        'source': doc.metadata.get('source', ''),
                        'page': doc.metadata.get('page', 0),
                        'image_url': doc.metadata.get('image_url', ''),
                        'content_type': doc.metadata.get('content_type', 'text'),
                        'timestamp': firestore.SERVER_TIMESTAMP
                    }
                    
                    self.collection.add(doc_data)
                    
                print(f"✅ Added {len(documents)} multimodal documents to Firestore")
                
            except Exception as e:
                print(f"Error adding multimodal documents: {e}")
                raise
    
    return FirestoreMultimodalStore(embedding_model, "multimodal_documents")

def create_vector_search_index():
    """Create Vector Search index in GCP"""
    try:
        config = get_gcp_config().get_vector_search_config()
        
        # Initialize Vertex AI
        aiplatform.init(
            project=get_gcp_config().project_id,
            location=get_gcp_config().location
        )
        
        # Create index endpoint
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name="rag-vector-endpoint",
            description="Vector search endpoint for RAG application"
        )
        
        print(f"✅ Vector Search index endpoint created: {index_endpoint.name}")
        return index_endpoint
        
    except Exception as e:
        print(f"Error creating vector search index: {e}")
        raise

#script for bigquery_client.py

import os
from typing import List, Dict, Any
from google.cloud import bigquery


class BigQueryClient:
    def __init__(self, project: str = None, dataset: str = None):
        self.project = project or os.getenv("SAFETY_GCP_PROJECT_ID")
        self.dataset = dataset or os.getenv("BIGQUERY_DATASET")
        if not self.dataset:
            raise ValueError("BIGQUERY_DATASET environment variable must be set")
        self.client = bigquery.Client(project=self.project)

    def table_ref(self, table_name: str):
        return f"{self.project}.{self.dataset}.{table_name}"

    def insert_rows(self, table_name: str, rows: List[Dict[str, Any]]):
        table_id = self.table_ref(table_name)
        errors = self.client.insert_rows_json(table_id, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")

    def run_query(self, query: str):
        job = self.client.query(query)
        return [dict(row) for row in job.result()]

    def get_ingestion_aggregation(self, table_name: str):
        # Group by file_name and aggregate file records
        table_id = self.table_ref(table_name)
        q = f"SELECT file_name, ARRAY_AGG(STRUCT(file_name AS file_name, file_type AS file_type, file_size AS file_size, source AS source, last_modified AS last_modified, ingested_at AS ingested_at, is_present AS is_present)) AS file_records FROM `{table_id}` GROUP BY file_name"
        return self.run_query(q)


class BigQueryByteStore:
    """Minimal byte store wrapper backed by BigQuery. Intended as a light
    replacement for a MongoDBByteStore used in the project. This provides a
    simple API the repo expects: init and insert/get operations.
    """
    def __init__(self, table_name: str = None):
        self.table = table_name or os.getenv("BIGQUERY_BYTESTORE_TABLE")
        self.bq = BigQueryClient()

    def add_documents(self, documents: List[Dict[str, Any]]):
        # documents are dictionaries representing rows; ensure required fields
        self.bq.insert_rows(self.table, documents)

    def find_by_id(self, doc_id: str):
        table_id = self.bq.table_ref(self.table)
        q = f"SELECT * FROM `{table_id}` WHERE doc_id = @doc_id LIMIT 1"
        job = self.bq.client.query(q, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id)]
        ))
        rows = [dict(r) for r in job.result()]
        return rows[0] if rows else None
    

# main.py

import os
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv

# ADK and Google Cloud Imports
import openai
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Your existing provider and utility functions
# from embedding_provider import vertex_ai_embed_model
# from vectordb_provider import get_vector_store_text


# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Configure the Generative AI client if needed (the Agent can also handle this)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Define the model you want the agent to use for its reasoning
AGENT_MODEL = LiteLlm(model="openai/gpt-4o")



async def retrieve_documents(query: str) -> Dict[str, Any]:
    """
    Retrieves relevant safety and compliance documents from the knowledge base.
    This tool should be used first to gather context before answering a question.
    """
    print(f"[Tool Call] retrieve_documents(query='{query}')")
    try:
        embedding_model = vertex_ai_embed_model()
        vector_store = get_vector_store_text("firestore", embedding_model)
        
        # Retrieve documents with their relevance scores
        docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
        
        if not docs_with_scores:
            return {"context": "No relevant documents were found.", "sources": []}
            
        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(docs_with_scores):
            source_info = {
                "file_name": doc.metadata.get('source', 'Unknown Source'),
                "relevance_score": f"{score:.3f}"
            }
            sources.append(source_info)
            # Create a clear context string for the LLM to read
            context_parts.append(f"Source: {source_info['file_name']}\nContent: {doc.page_content}\n---")
            
        return {
            "context": "\n".join(context_parts),
            "sources": sources
        }
    except Exception as e:
        print(f"Error in retrieve_documents tool: {e}")
        return {"context": f"An error occurred during document retrieval: {e}", "sources": []}

async def generate_follow_up_questions(original_question: str, generated_answer: str) -> List[str]:
    """
    Generates 3 contextually relevant follow-up questions...
    """
    print(f"[Tool Call] generate_follow_up_questions(...)")
    try:
        # Instantiate the async OpenAI client
        client = openai.AsyncOpenAI()
        
        prompt = f"""Based on the user's question and the AI's response, generate 3 relevant follow-up questions.
User's Original Question: {original_question}
AI's Response: {generated_answer}
Generate 3 follow-up questions. Format the response as a Python list of strings. Example: ["Question 1?", "Question 2?", "Question 3?"]"""

        # Make the API call to OpenAI
        response = await client.chat.completions.create(
            model="gpt-4o", # You can use a faster model like gpt-3.5-turbo if you prefer
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        
        # This parsing is simple and assumes the model returns a string like '["q1", "q2"]'
        questions = [q.strip().strip("'\"") for q in content.strip()[1:-1].split(',')]
        return questions[:3]
        
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        return []



root_agent = Agent(
    name="safety_compliance_agent",
    model=AGENT_MODEL,
    description="An agent that answers safety and compliance questions by first searching a knowledge base.",
    
    # The instructions are the most critical part. They tell the agent *how* to use the tools.
    instruction=(
        "You are an expert AI assistant specializing in safety and compliance protocols. "
        "Your goal is to provide accurate and helpful answers based on a set of internal documents. "
        "Follow this workflow strictly:"
        "1. When the user asks a question, your **first action** must be to use the `retrieve_documents` tool to find relevant information. "
        "2. Analyze the 'context' returned by the tool. Based **only** on this information, formulate a comprehensive and clear answer to the user's question. "
        "3. Do not cite document IDs or file names in your final answer. Synthesize the information into a helpful response. If the context is insufficient, state that you could not find the information. "
        "4. After you have formulated your final answer, do not display it yet. Your **next action** is to call the `generate_follow_up_questions` tool. Pass the user's original question and the complete answer you just formulated to this tool. "
        "5. Finally, present the complete response to the user: first your detailed answer, and then the suggested follow-up questions under a clear heading."
    ),
    
    # We provide the functions directly to the agent.
    tools=[retrieve_documents, generate_follow_up_questions]
    
    # Enable streaming for a better user experience
    
)
    






from google.oauth2 import service_account
from vertexai import agent_engines
from vertexai.agent_engines import AdkApp
from vertexai.preview import reasoning_engines
import vertexai
import json
from dotenv import load_dotenv
load_dotenv()






gcp_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")



PROJECT_ID = os.environ.get("GCP_PROJECT_ID_1")
LOCATION = os.environ.get("LOCATION")
STAGING_BUCKET = "gs://"+os.environ.get("GCS_BUCKET_NAME")

print(PROJECT_ID,LOCATION,STAGING_BUCKET)
credentials = service_account.Credentials.from_service_account_file(gcp_credentials)

# storage.Client(project=PROJECT_ID, credentials=credentials)
# Initialize Vertex AI with explicit credentials if available
vertexai.init(project=PROJECT_ID, location=LOCATION,staging_bucket=STAGING_BUCKET,credentials=credentials)


app = AdkApp(agent=root_agent)




remote_app = agent_engines.create(
                agent_engine=app,
                # extra_packages=["embedding_provider.py","bigquery_client.py","gcp_config.py","vectordb_provider.py"],
                requirements=["google-adk",
                                "vertexai",
                                "cloudpickle",
                                "google-cloud-aiplatform[adk]",
                                "pydantic",

                                # For using LiteLLM with OpenAI
                                "litellm",
                                "openai",

                                "langchain-core",
                                "langchain-google-firestore",
                                "langchain-google-vertexai",
                                
                                # For your vectordb_provider and embedding_provider
                                "google-cloud-firestore",
                                "google-auth",  # Good practice to include for authentication
                                "python-dotenv"],
                display_name="safety Agent v2",
                
                description="ADK Agent to help users to information about safely rules and regulations from documents",
                env_vars={
                    "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
                    "GCP_LOCATION": os.getenv("GCP_LOCATION"),
                    "AUTH_URI":os.getenv("AUTH_URI"),
                    "SAFETY_GCP_PROJECT_ID":os.getenv("SAFETY_GCP_PROJECT_ID"),
                    "VERTEX_AI_LOCATION":os.getenv("VERTEX_AI_LOCATION"),
                    "BIGQUERY_DATASET":os.getenv("BIGQUERY_DATASET"),
                    "BIGQUERY_USERS_TABLE":"users",
                    "BIGQUERY_RATINGS_TABLE":"message_ratings",
                    "BIGQUERY_BYTESTORE_TABLE":"byte_store",

                    "FIRESTORE_COLLECTION":"safety_documents",
                    "FIRESTORE_DATABASE":os.getenv("FIRESTORE_DATABASE"),
                    "FIRESTORE_DOCUMENT_COLLECTION":"safety_documents",

                    "VECTOR_K":"5",
                    "MULTIMODAL_K":"5",
                    "THRESHOLD":"0.6",
                    "OPENAI_API_KEY":os.getenv("OPENAI_API_KEY"),
                    "GOOGLE_GENAI_USE_VERTEXAI":"0",
                    "type": "service_account", # This field is required by from_service_account_info
                    "project_id": os.getenv("GCP_PROJECT_ID"), # Use the project_id from the class
                    "private_key_id": os.getenv("private_key_id"),
                    "private_key": os.getenv("private_key"),
                    "client_email": os.getenv("client_email"),
                    "client_id": os.getenv("client_id"),
                    "auth_uri": os.getenv("auth_uri"),
                    "token_uri": os.getenv("token_uri"),
                    "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
                    "client_x509_cert_url": os.getenv("client_x509_cert_url"),
                    "universe_domain": os.getenv("universe_domain")
                                                },
                                        

            )
 
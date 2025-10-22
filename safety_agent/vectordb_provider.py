from google.cloud import firestore
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
from gcp_config import gcp_config
from bigquery_client import BigQueryClient
import numpy as np
from typing import List, Dict, Any
import uuid
from langchain_core.documents import Document
import time
import os
import json
from datetime import datetime

# Initialize GCP clients
firestore_client = gcp_config.firestore_client
storage_client = gcp_config.storage_client

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
    config = gcp_config.get_vector_search_config()
    
    class VertexAIVectorStore:
        def __init__(self, embedding_model, index_id=None):
            self.embedding_model = embedding_model
            self.index_id = index_id or config['index_id']
            self.project_id = gcp_config.project_id
            self.location = gcp_config.location
            
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
                    doc_ref = firestore_client.collection('documents').document(neighbor.id)
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
    config = gcp_config.get_vector_search_config()
    
    class VertexAIMultimodalStore:
        def __init__(self, embedding_model, index_id=None):
            self.embedding_model = embedding_model
            self.index_id = index_id or f"{config['index_id']}_multimodal"
            self.project_id = gcp_config.project_id
            self.location = gcp_config.location
            
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
                    doc_ref = firestore_client.collection('multimodal_documents').document(neighbor.id)
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
    config = gcp_config.get_firestore_config()
    
    class FirestoreVectorStore:
        def __init__(self, embedding_model, collection_name="documents"):
            self.embedding_model = embedding_model
            self.collection = firestore_client.collection(collection_name)
        
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
    config = gcp_config.get_firestore_config()
    
    class FirestoreMultimodalStore:
        def __init__(self, embedding_model, collection_name="multimodal_documents"):
            self.embedding_model = embedding_model
            self.collection = firestore_client.collection(collection_name)
        
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
        config = gcp_config.get_vector_search_config()
        
        # Initialize Vertex AI
        aiplatform.init(
            project=gcp_config.project_id,
            location=gcp_config.location
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
# main.py - ALL-IN-ONE SCRIPT

# ==============================================================================
# 1. IMPORTS - All imports from all files are collected at the top
# ==============================================================================
import os
import asyncio
import json
from typing import Dict, List, Any
from datetime import datetime

# Environment and Credentials
from dotenv import load_dotenv
from google.oauth2 import service_account

# Google Cloud Services
from google.cloud import aiplatform, firestore, storage, bigquery
import vertexai
from vertexai.language_models import TextEmbeddingModel
from langchain_core.documents import Document

# ADK and Agent Framework
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from vertexai import agent_engines
from vertexai.agent_engines import AdkApp

# Other Libraries
import openai
import numpy as np

# ==============================================================================
# 2. CONFIGURATION & INITIALIZATION
# ==============================================================================
load_dotenv()

# --- BigQuery Client Class (from bigquery_client.py) ---
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
# --- GCP Config Class (from gcp_config.py) ---
class GCPConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # This check prevents re-initializing the singleton on subsequent calls
        if hasattr(self, "project_id"):
            return

        # All initialization logic now lives in __init__
        self.project_id = os.getenv("SAFETY_GCP_PROJECT_ID", "safety-agent-469708")
        self.location = os.getenv("GCP_LOCATION", "us-central1")
        print(f"🔧 GCP Config Initializing - Project: {self.project_id}, Location: {self.location}")
        self._init_clients()

    def _get_credentials(self) -> service_account.Credentials:
        # NOTE: Using replace('\\n', '\n') is crucial when loading from .env files
        private_key_env = os.getenv("private_key")
        if not private_key_env:
            raise ValueError("The 'private_key' environment variable is not set.")
        
        service_account_info = {
            "type": "service_account",
            "project_id": self.project_id,
            "private_key_id": os.getenv("private_key_id"),
            "private_key": private_key_env.replace('\\n', '\n'),
            "client_email": os.getenv("client_email"),
            "client_id": os.getenv("client_id"),
            "auth_uri": os.getenv("auth_uri"),
            "token_uri": os.getenv("token_uri"),
            "auth_provider_x509_cert_url": os.getenv("auth_provider_x509_cert_url"),
            "client_x509_cert_url": os.getenv("client_x509_cert_url"),
            "universe_domain": os.getenv("universe_domain")
        }
        return service_account.Credentials.from_service_account_info(service_account_info)

    def _init_clients(self):
        try:
            credentials = self._get_credentials()
            # Initialize Vertex AI globally for other functions that might need it
            vertexai.init(project=self.project_id, location=self.location, credentials=credentials)
            # Initialize specific clients for the config object
            self.firestore_client = firestore.Client(project=self.project_id, credentials=credentials)
            self.storage_client = storage.Client(project=self.project_id, credentials=credentials)
            print("✅ GCP clients initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing GCP clients: {e}")
            raise

    def get_firestore_config(self):
        return {"collection": os.getenv("FIRESTORE_COLLECTION", "safety_documents")}

# Global function to access the singleton config instance
def get_gcp_config():
    # This will now correctly create and initialize the instance once
    return GCPConfig()

# ==============================================================================
# 3. PROVIDER FUNCTIONS (from embedding_provider.py and vectordb_provider.py)
# ==============================================================================

# --- Embedding Provider Functions ---
def vertex_ai_embed_model(model: str = None):
    try:
        model_name = model or os.getenv("EMBEDDING_MODEL_NAME") or "text-embedding-004"
        # No need to call aiplatform.init here, it's handled by GCPConfig
        embedding_model = TextEmbeddingModel.from_pretrained(model_name)
        return embedding_model
    except Exception as e:
        print(f"⚠️ Error initializing Vertex AI embedding model '{model_name}': {e}")
        return None

# --- Vector DB Provider Functions ---
def _get_firestore_vector_store_text(embedding_model):
    config = get_gcp_config().get_firestore_config()
    
    class FirestoreVectorStore:
        def __init__(self, embedding_model, collection_name="documents"):
            self.embedding_model = embedding_model
            self.collection = get_gcp_config().firestore_client.collection(collection_name)
        
        def similarity_search_with_score(self, query: str, k: int = 5):
            try:
                query_embedding = self.embedding_model.get_embeddings([query])[0].values
                documents_stream = self.collection.stream()
                
                docs_and_scores = []
                for doc in documents_stream:
                    doc_data = doc.to_dict()
                    if 'embedding' in doc_data:
                        doc_embedding = np.array(doc_data['embedding'])
                        similarity = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
                        document = Document(
                            page_content=doc_data.get('text', ''),
                            metadata={'source': doc_data.get('source', ''), 'page': doc_data.get('page', 0), 'doc_id': doc.id}
                        )
                        docs_and_scores.append((document, similarity))
                
                docs_and_scores.sort(key=lambda x: x[1], reverse=True)
                return docs_and_scores[:k]
            except Exception as e:
                print(f"Error in similarity search: {e}")
                return []

    return FirestoreVectorStore(embedding_model, config['collection'])

def get_vector_store_text(vector_store_provider: str, embedding_model):
    if vector_store_provider.lower() == "firestore":
        return _get_firestore_vector_store_text(embedding_model)
    else:
        raise ValueError(f"Unknown vector store: {vector_store_provider}")

# ==============================================================================
# 4. AGENT TOOL DEFINITIONS
# ==============================================================================
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
        docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
        
        if not docs_with_scores:
            return {"context": "No relevant documents were found.", "sources": []}
            
        context_parts = []
        sources = []
        for i, (doc, score) in enumerate(docs_with_scores):
            # *** FIX for empty file_name ***
            # Ensure you have a 'source' field in your Firestore documents with the filename.
            file_name = doc.metadata.get('source', 'Unknown Source')
            source_info = {"file_name": file_name, "relevance_score": f"{score:.3f}"}
            sources.append(source_info)
            context_parts.append(f"Source: {file_name}\nContent: {doc.page_content}\n---")
            
        return {"context": "\n".join(context_parts), "sources": sources}
    except Exception as e:
        print(f"Error in retrieve_documents tool: {e}")
        return {"context": f"An error occurred during document retrieval: {e}", "sources": []}

async def generate_follow_up_questions(original_question: str, generated_answer: str) -> List[str]:
    """
    Generates 3 contextually relevant follow-up questions...
    """
    print(f"[Tool Call] generate_follow_up_questions(...)")
    try:
        client = openai.AsyncOpenAI()
        prompt = f"""Based on the user's question and the AI's response, generate 3 relevant follow-up questions.
User's Original Question: {original_question}
AI's Response: {generated_answer}
Generate 3 follow-up questions. Format the response as a Python list of strings. Example: ["Question 1?", "Question 2?", "Question 3?"]"""
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        # A more robust way to parse the list-like string from the LLM
        questions = json.loads(content.replace("'", '"'))
        return questions[:3]
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        return []

# ==============================================================================
# 5. AGENT DEFINITION & DEPLOYMENT SCRIPT
# ==============================================================================
root_agent = Agent(
    name="safety_compliance_agent",
    model=AGENT_MODEL,
    description="An agent that answers safety and compliance questions by first searching a knowledge base.",
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
    tools=[retrieve_documents, generate_follow_up_questions]
)

# --- Deployment Logic ---
# This part of the script will only run when you execute `python main.py`
if __name__ == "__main__":
    PROJECT_ID = os.environ.get("GCP_PROJECT_ID_1")
    LOCATION = os.environ.get("LOCATION")
    STAGING_BUCKET = "gs://" + os.environ.get("GCS_BUCKET_NAME")
    gcp_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    print(f"Deploying to Project: {PROJECT_ID}, Location: {LOCATION}, Bucket: {STAGING_BUCKET}")
    
    credentials = service_account.Credentials.from_service_account_file(gcp_credentials_path)
    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET, credentials=credentials)

    app = AdkApp(agent=root_agent)

    remote_app = agent_engines.create(
        agent_engine=app,
        # extra_packages is now empty because all code is in this file.
        requirements=[
            "google-adk",
            "vertexai",
            "cloudpickle",
            "google-cloud-aiplatform[adk]",
            "pydantic",
            "litellm",
            "openai",
            "langchain-core",
            "langchain-google-firestore",
            "langchain-google-vertexai",
            "google-cloud-firestore",
            "google-auth",
            "python-dotenv"
        ],
        display_name="Safety Agent v5",
        description="ADK Agent in a single file to answer safety questions from documents.",
        env_vars={
            # ... keep all your existing env_vars ...
            "SAFETY_GCP_PROJECT_ID": os.getenv("SAFETY_GCP_PROJECT_ID"),
            "GCP_LOCATION": os.getenv("GCP_LOCATION"),
            "BIGQUERY_DATASET": os.getenv("BIGQUERY_DATASET"),
            "FIRESTORE_COLLECTION": os.getenv("FIRESTORE_COLLECTION", "safety_documents"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
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
    )
    print(f"Agent deployed successfully: {remote_app.name}")
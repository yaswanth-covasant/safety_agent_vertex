# main.py

import os
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv

# ADK and Google Cloud Imports
import openai
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
import google.generativeai as genai

# Your existing provider and utility functions
from embedding_provider import vertex_ai_embed_model
from vectordb_provider import get_vector_store_text


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
                extra_packages=["embedding_provider.py","bigquery_client.py","gcp_config.py","vectordb_provider.py"],
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
                display_name="safety Agent",
                
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
 
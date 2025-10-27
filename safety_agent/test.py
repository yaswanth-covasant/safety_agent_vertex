
from vertexai import agent_engines
from google.cloud import aiplatform
import asyncio

PROJECT_ID = "gcp-agents-personal"
LOCATION = "us-central1"
AGENT_ID = "projects/208578852509/locations/us-central1/reasoningEngines/954679558118834176"


async def query_agent_async():
    """Query agent using async methods."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    agent = agent_engines.get(AGENT_ID)
    
    user_id = "user_1"
    
    # Create session
    session = await agent.async_create_session(user_id=user_id)
    session_id = session.get("id") or session.get("session_id") or session.get("name")
    print(f"Session ID: {session_id}\n")
    
    messages = [
        
        "How Can OSHA and NIOSH Help?"
    ]
    
    for message in messages:
        print(f">>> {message}")
        async for event in agent.async_stream_query(
            message=message,
            user_id=user_id,
            session_id=session_id
        ):
            print(event)
            print("-" * 50)
    
    # Get final session
   


if __name__ == "__main__":
    asyncio.run(query_agent_async())
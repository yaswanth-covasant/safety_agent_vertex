from vertexai import agent_engines

PROJECT_ID = "gcp-agents-personal"
LOCATION = "us-central1"


def query_the_agent():
    """Sends a single query to the deployed agent to generate a log entry."""
    print("--- STEP 1: Sending query to the agent ---")
    try:
        agent =agent_engines.get('projects/208578852509/locations/us-central1/reasoningEngines/2213435653968887808')
        session = agent.create_session(user_id="user_1")
        messages=["hi","How often should PPE be inspected and maintained for chemical handling to ensure it remains effective?"]
        for message in messages:
            quer = agent.stream_query(user_id="user_1",session_id=session["id"],message=message)
            for i in quer:
                print(i)
                print("-------------------")
                
                
        
        session=agent.get_session(user_id="sai",session_id=session["id"])
        for event in session['events']:
            print("---------------new event----------------------------------------")
            print(event)

    except Exception as e:
        print(f"ERROR: Failed to query the agent. {e}")
        print("Please check your agent ID, permissions, and ensure the agent is deployed.\n")
        return False
    return True


if __name__ == "__main__":
    query_the_agent()
   
    
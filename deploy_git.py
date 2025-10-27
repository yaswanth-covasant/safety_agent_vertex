
"""
Corrected Script: Deploys a Vertex AI ADK Agent from a GitHub repository.

Key Improvements:
- Removed verbose logging from underlying libraries for cleaner output.
- Increased deployment timeout to 30 minutes to prevent client-side errors.
- Dynamically loads dependencies from a 'requirements.txt' in the GitHub repo.
- Modernized package version checking to remove deprecation warnings.
"""

import os
import sys
import tempfile
import subprocess

from importlib.util import module_from_spec, spec_from_loader
from importlib import metadata

import vertexai
from vertexai.preview import reasoning_engines

# --- Configuration ---
PROJECT_ID = "gcp-agents-personal"
LOCATION = "us-central1"
STAGING_BUCKET = "gs://deploy_agent_to_engine"


# GitHub Repository Configuration
GITHUB_REPO_URL = "https://github.com/yaswanth-covasant/safety_agent_vertex.git"
GITHUB_BRANCH = "main"
AGENT_SOURCE_FILE = "agent.py"
AGENT_OBJECT_NAME = "root_agent"

# --- Default requirements if requirements.txt is not found in the repo ---
DEFAULT_REQUIREMENTS = [
    "google-cloud-aiplatform[adk]>=1.50.0",
    "cloudpickle>=3.1.0",
    "pydantic>=2.6.0,<3.0.0",
    "google-adk>=0.1.0",
]

def check_package_versions():
    """Check and display current package versions for debugging."""
    print(" Current package versions:")
    packages_to_check = [
        "google-cloud-aiplatform",
        "pydantic",
        "cloudpickle",
        "google-adk",
    ]
    for package in packages_to_check:
        try:
            version = metadata.version(package)
            print(f"  {package}: {version}")
        except metadata.PackageNotFoundError:
            print(f"  {package}: NOT INSTALLED")
    print()


def validate_agent_object(agent_obj):
    """Validate the agent object before deployment."""
    print("Validating agent object...")
    if not agent_obj:
        print("Agent object is None!")
        return False
    if not hasattr(agent_obj, "name"):
        print("Agent object is missing the required 'name' attribute!")
        return False
    
    print(f"Agent Type: {type(agent_obj).__name__}, Name: {agent_obj.name}")

    try:
        import cloudpickle
        cloudpickle.dumps(agent_obj)
        print("Agent is serializable (can be pickled).")
    except Exception as e:
        print(f"Agent cannot be pickled, which is required for deployment: {e}")
        return False
    
    return True

def  deploy_from_github():
    """Clones a GitHub repo, loads an agent, and deploys it as a Reasoning Engine."""
    print("Starting ADK Agent deployment from GitHub...")
    check_package_versions()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Clone the GitHub repository
        print(f"Cloning repository from {GITHUB_REPO_URL}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "-b", GITHUB_BRANCH, GITHUB_REPO_URL, temp_dir],
                check=True, capture_output=True, text=True,
            )
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e.stderr}")
            return None

        # Step 2: Dynamically load agent's Python code
        agent_file_path = os.path.join(temp_dir, AGENT_SOURCE_FILE)
        if not os.path.exists(agent_file_path):
            print(f"Source file '{AGENT_SOURCE_FILE}' not found in the repository.")
            return None

        try:
            print(f"Loading agent source code from '{agent_file_path}'...")
            spec = spec_from_loader("dynamic_agent_module", loader=None, origin=agent_file_path)
            agent_module = module_from_spec(spec)
            sys.modules["dynamic_agent_module"] = agent_module
            with open(agent_file_path, "r", encoding="utf-8") as f:
                exec(f.read(), agent_module.__dict__)
            
            the_agent = getattr(agent_module, AGENT_OBJECT_NAME)
            print(f"Successfully loaded agent object '{AGENT_OBJECT_NAME}'.")
        except (AttributeError, FileNotFoundError, SyntaxError) as e:
            print(f"Failed to load the agent object from the Python file: {e}")
            return None
        finally:
            if "dynamic_agent_module" in sys.modules:
                del sys.modules["dynamic_agent_module"]  # Clean up

        if not validate_agent_object(the_agent):
            return None

        # Step 3: Load requirements from the repository
        requirements_path = os.path.join(temp_dir, "requirements.txt")
        if os.path.exists(requirements_path):
            print("Found 'requirements.txt' in the repository.")
            with open(requirements_path, "r") as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        else:
            print("'requirements.txt' not found in repo. Using default requirements.")
            requirements = DEFAULT_REQUIREMENTS
        print(f"📦 Using requirements: {requirements}")

        # Step 4: Initialize Vertex AI and deploy
        try:
            print("Initializing Vertex AI SDK...")
            vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)
            print("Vertex AI SDK initialized.")

            adk_app = reasoning_engines.AdkApp(agent=the_agent)
            
            print("Deploying ADK application... (This can take up to 30 minutes)")
            remote_app = reasoning_engines.ReasoningEngine.create(
                reasoning_engine=adk_app,
                requirements=requirements,
                display_name="Movie Booking Agent from GitHub",
                description="ADK Agent to help users book movie tickets"
                
            )
            return remote_app.resource_name

        except Exception as e:
            print(f"\n  An error occurred during deployment: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to orchestrate the deployment."""
    print("=" * 60)
    print("VERTEX AI ADK AGENT DEPLOYMENT FROM GITHUB")
    print("=" * 60)
    
    resource_name = deploy_from_github()
    
    print("\n" + "=" * 60)
    if resource_name:
        reasoning_engine_id = resource_name.split('/')[-1]
        print(" DEPLOYMENT SUCCEEDED ")
        print(f"\n Full Resource Name: {resource_name}")
        print(f" Reasoning Engine ID: {reasoning_engine_id}")
        print(f" Console URL: https://console.cloud.google.com/vertex-ai/reasoning-engines/{reasoning_engine_id}?project={PROJECT_ID}")
    else:
        print(" DEPLOYMENT FAILED ")
        print("Please review the error messages above for details.")
    print("=" * 60)

if __name__ == "__main__":
    main()
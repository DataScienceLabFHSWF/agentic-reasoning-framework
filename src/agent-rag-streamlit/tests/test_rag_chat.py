import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_utils.agentic_rag_chat import create_rag_chat

CHROMA_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"
PROCESSED_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"

print("Initializing RAG Chat System...")
print("This may take a moment as we load all models into memory...")

# Create RAG chat with specific models and relevance threshold
rag_chat = create_rag_chat(
    chroma_dir=CHROMA_DIR,
    processed_dir=PROCESSED_DIR,
    router_model="gpt-oss:20b",        # Reasoning model for routing
    summarizer_model="llama3.1:latest", # Summarization model
    general_model="llama3.1:latest",     # General conversation model
    relevance_threshold=0.5              # Minimum relevance score for using docs
)

print("All models loaded! Starting chat...")

# Start interactive chat
rag_chat.start_chat()
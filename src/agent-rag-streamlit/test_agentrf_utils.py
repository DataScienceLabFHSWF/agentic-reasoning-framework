import os
import sys

# 1. Setup path to allow importing modules from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentrf_utils import ingest_file_to_chroma

if __name__ == "__main__":
    # 2. Define absolute paths based on project structure
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_path = "/home/rrao/projects/agents/agentic-reasoning-framework/src/agent-rag-streamlit/data/uvu.pdf"
    existing_db_path = os.path.join(base_dir, "agent-rag-streamlit/chroma_db")

    print(f"Ingesting '{os.path.basename(pdf_path)}' into existing DB: {existing_db_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 3. Run Ingestion
    success = ingest_file_to_chroma(file_path=pdf_path, persist_directory=existing_db_path)

    print("Success! Document added." if success else "Failed.")

    from agent_utils.agentic_rag_chat import create_rag_chat

    CHROMA_DIR = "../chroma_db"
    PROCESSED_DIR = "../processed_files"

    print("Initializing RAG Chat System...")
    print("This may take a moment as we load all models into memory...")

    # Create RAG chat with specific models and relevance threshold
    rag_chat = create_rag_chat(chroma_dir = existing_db_path, processed_dir = PROCESSED_DIR, 
                                router_model = "mistral-small3.2", 
                                reasoning_model="mistral-small3.2",
                                summarizer_model = "mistral:v0.3", 
                                general_model = "mistral:v0.3",
                                final_answer_model="mistral:v0.3", 
                                relevance_threshold = 0.2)

    print("All models loaded! Starting chat...")

    # Start interactive chat
    rag_chat.start_chat()
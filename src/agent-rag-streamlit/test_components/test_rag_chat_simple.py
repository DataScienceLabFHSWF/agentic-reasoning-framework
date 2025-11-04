import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from agent_utils.workflow import create_rag_chat
CHROMA_DIR = "../chroma_db"
PROCESSED_DIR = "../processed_files"


# Default: Ollama with llama3.1:latest
rag_chat = create_rag_chat(
    chroma_dir=CHROMA_DIR,
    processed_dir=PROCESSED_DIR
)

# Start infinite chat
rag_chat.start_chat()
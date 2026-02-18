import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from rag_utils.retriever import hybrid_retrieve

CHROMA_DIR = os.environ.get("CHROMA_DIR", "/home/rrao/projects/agents/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/home/rrao/projects/agents/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files")

@tool
def retrieve_documents(query: str) -> str:
    """Search the nuclear engineering and safety knowledge base for facts, technical specifications, or safety reports."""
    docs = hybrid_retrieve(query=query, chroma_dir=CHROMA_DIR, processed_dir=PROCESSED_DIR, k=3)
    if not docs:
        return "Keine relevanten Dokumente gefunden."
    return "\n\n".join(
        f"--- {doc.metadata.get('filename', f'Doc {i}')} ---\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )

llm = ChatOllama(model="qwen3:30b", temperature=0.0)

agent = create_agent(
    model=llm,
    tools=[retrieve_documents],
    system_prompt=(
        "Du bist ein Experte fuer deutsche Kerntechnik und Nuklearsicherheit. Antworte immer auf Deutsch. "
        "Nutze das 'retrieve_documents' Tool, um Fakten nachzuschlagen. "
        "Fuehre bei Bedarf mehrere Suchen mit abgewandelten Begriffen durch, bevor du eine finale Antwort gibst."
    )
)

if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        stream = agent.stream({"messages": [{"role": "user", "content": user_input}]})
        for chunk in stream:
            # 'chunk' is a dictionary showing which node just ran (e.g., 'agent' or 'tools')
            for node_name, state_update in chunk.items():
                print("\n--- AGENT UPDATE ---")
                print(node_name.upper())
                print(state_update)

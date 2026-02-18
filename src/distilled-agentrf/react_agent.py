import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.agents import create_agent          # langchain >= 1.0
from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from rag_utils.retriever import hybrid_retrieve

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = os.environ.get("CHROMA_DIR",    "/home/rrao/projects/agents/agentic-reasoning-framework/src/distilled-agentrf/chroma_db")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/home/rrao/projects/agents/agentic-reasoning-framework/src/distilled-agentrf/processed_files")

# ── Tool ──────────────────────────────────────────────────────────────────────
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

# ── Prompts ───────────────────────────────────────────────────────────────────
SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Du bist ein Experte für deutsche Kerntechnik und Nuklearsicherheit. "
     "Fasse die folgende Rohantwort in klarem, verständlichem Deutsch zusammen. "
     "Behalte alle wichtigen Fakten, Zahlen und Fachbegriffe bei."),
    ("human", "Frage: {query}\n\nRohantwort:\n{raw_answer}"),
])

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extrahiere die kürzestmögliche Antwort (ein Wort, eine Zahl oder eine Entität) "
     "aus der zusammengefassten Antwort. Kein zusätzlicher Text."),
    ("human", "Frage: {query}\n\nZusammengefasste Antwort:\n{summarized_answer}"),
])

# ── Models ────────────────────────────────────────────────────────────────────
REACT_MODEL = ChatOllama(model="qwen3:30b",    temperature=0.0)
SUMMARY_LLM = ChatOllama(model="mistral:v0.3", temperature=0.0)
FINAL_LLM   = ChatOllama(model="mistral:v0.3", temperature=0.0)

# ── Agent ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Du bist ein Experte fuer deutsche Kerntechnik und Nuklearsicherheit. Antworte immer auf Deutsch. "
    "Nutze das 'retrieve_documents' Tool, um Fakten nachzuschlagen. "
    "Fuehre bei Bedarf mehrere Suchen mit abgewandelten Begriffen durch, bevor du eine finale Antwort gibst."
)

agent = create_agent(
    model=REACT_MODEL,
    tools=[retrieve_documents],
    system_prompt=SYSTEM_PROMPT,
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run(user_input: str) -> dict:
    raw_answer = ""

    # 1. ReAct agent — stream intermediate steps, capture final AI message
    print("\n[Agent]", flush=True)
    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    ):
        latest = chunk["messages"][-1]
        if isinstance(latest, AIMessage):
            if latest.tool_calls:
                print(f"  → calling tool: {[tc['name'] for tc in latest.tool_calls]}", flush=True)
            elif latest.content:
                print(f"  {latest.content}", flush=True)
                raw_answer = latest.content   # last AI text = final reasoning answer

    # 2. Summarizer
    summarized = (SUMMARIZER_PROMPT | SUMMARY_LLM).invoke(
        {"query": user_input, "raw_answer": raw_answer}
    ).content

    # 3. Final answer
    final = (FINAL_ANSWER_PROMPT | FINAL_LLM).invoke(
        {"query": user_input, "summarized_answer": summarized}
    ).content

    return {"raw": raw_answer, "summarized": summarized, "final": final}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        out = run(user_input)
        print("\n[Summarized]", out["summarized"])
        print("\n[Final]",      out["final"])
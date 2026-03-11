import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from rag_utils.retriever import hybrid_retrieve

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = os.environ.get("CHROMA_DIR",    "/home/rrao/projects/agents/agentic-reasoning-framework/src/distilled-agentrf/chroma_db")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/home/rrao/projects/agents/agentic-reasoning-framework/src/distilled-agentrf/processed_files")

# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def retrieve_documents(query: str) -> str:
    """Search the nuclear engineering and safety knowledge base."""
    docs = hybrid_retrieve(query=query, chroma_dir=CHROMA_DIR, processed_dir=PROCESSED_DIR, k=5)
    if not docs:
        return "Keine relevanten Dokumente gefunden."
    return "\n\n".join(
        f"--- {doc.metadata.get('filename', f'Doc {i}')} ---\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )

# ── Prompts ───────────────────────────────────────────────────────────────────
SUMMARIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Beantworte die Frage ausschließlich basierend auf dem gegebenen Kontext. Antworte auf Deutsch."),
    ("human", "Frage: {query}\n\nKontext:\n{raw_answer}"),
])

FINAL_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extrahiere die kürzestmögliche Antwort (ein Wort, eine Zahl oder Entität). Kein zusätzlicher Text."),
    ("human", "Frage: {query}\n\nAntwort:\n{summarized_answer}"),
])

# ── Models ────────────────────────────────────────────────────────────────────
REACT_MODEL = ChatOllama(model="mistral-small3.2:latest", temperature=0.0)
SUMMARY_LLM = ChatOllama(model="mistral:v0.3", temperature=0.0)
FINAL_LLM   = ChatOllama(model="mistral:v0.3", temperature=0.0)

# ── Agent ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "Nutze retrieve_documents um Fragen zu beantworten. Antworte auf Deutsch."

agent = create_agent(
    model=REACT_MODEL,
    tools=[retrieve_documents],
    system_prompt=SYSTEM_PROMPT,
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run(user_input: str) -> dict:
    print("\n[Agent]", flush=True)
    try:
        raw_parts = []
        for chunk in agent.stream({"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config={"recursion_limit": 6},
        ):
            latest = chunk["messages"][-1]
            print(f"  [Agent Update] {latest.type}", flush=True)
            print(f"  [Full Message] {latest.content}", flush=True)
            if isinstance(latest, AIMessage):
                if latest.tool_calls:
                    for tc in latest.tool_calls:
                        query = tc.get('args', {}).get('query', tc.get('args', {}))
                        print(f"[TOOL CALL]  → [{tc['name']}] query: \"{query}\"", flush=True)
                elif latest.content:
                    print(f"  {latest.content}")
                    raw_parts.append(latest.content)
            elif isinstance(latest, ToolMessage):
                preview = latest.content[:500] + "..." if len(latest.content) > 500 else latest.content
                print(f"  [Tool Result]\n{preview}", flush=True)
                raw_parts.append(f"[Retrieved Documents]\n{latest.content}")
    except GraphRecursionError:
        print("  [recursion limit reached]", flush=True)
        if not raw_parts:
            raw_parts.append("Keine ausreichenden Informationen gefunden.")

    raw_answer = "\n".join(raw_parts)

    summarized = (SUMMARIZER_PROMPT | SUMMARY_LLM).invoke(
        {"query": user_input, "raw_answer": raw_answer}
    ).content

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
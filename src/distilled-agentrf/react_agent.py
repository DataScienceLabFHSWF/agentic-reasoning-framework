import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain.agents import create_agent          # langchain >= 1.0
from langchain.tools import tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from rag_utils.retriever import hybrid_retrieve
from langchain_core.messages import AIMessage, ToolMessage


# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_DIR    = os.environ.get("CHROMA_DIR",    "/home/rrao/projects/agents/agentic-reasoning-framework/src/distilled-agentrf/chroma_db")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/home/rrao/projects/agents/agentic-reasoning-framework/src/distilled-agentrf/processed_files")

# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def retrieve_documents(query: str) -> str:
    """Search the nuclear engineering and safety knowledge base for facts, technical specifications, or safety reports."""
    docs = hybrid_retrieve(query=query, chroma_dir=CHROMA_DIR, processed_dir=PROCESSED_DIR, k=5)
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
REACT_MODEL = ChatOllama(model="qwen3:30b",    temperature=0.0) #
SUMMARY_LLM = ChatOllama(model="mistral:v0.3", temperature=0.0)
FINAL_LLM   = ChatOllama(model="mistral:v0.3", temperature=0.0)

# ── Agent ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
    Du bist ein Experte für deutsche Kerntechnik und Nuklearsicherheit. Antworte immer auf Deutsch.

Wenn die Frage Fakten, Normen, Grenzwerte, Definitionen, Ereignisse, Institutionen oder konkrete Behauptungen enthält, nutze zuerst das Tool „retrieve_documents", um die relevanten Quellenstellen zu finden.
Wenn du nach der ersten Suche nicht genügend Informationen hast, um die Frage zu beantworten, formuliere eine neue, präzisere Suchanfrage und verwende das Tool „retrieve_documents" erneut, um weitere Informationen zu finden.
Wenn du genügend Informationen hast, um die Frage zu beantworten, gib eine klare und präzise Antwort auf Deutsch. Beziehe dich dabei auf die gefundenen Dokumente und zitiere sie, wenn möglich.
"""

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
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
            config={"recursion_limit": 6},  # ~5 tool calls
        ):
            latest = chunk["messages"][-1]
            if isinstance(latest, AIMessage):
                if latest.tool_calls:
                    for tc in latest.tool_calls:
                        name = tc['name']
                        args = tc.get('args', {})
                        query = args.get('query', args)  # fallback to full args if no 'query' key
                        print(f"  → [{name}] query: \"{query}\"", flush=True)
                elif latest.content:
                    print(f"  {latest.content}")
                    raw_parts.append(latest.content)
            elif isinstance(latest, ToolMessage):
                raw_parts.append(f"[Retrieved Documents]\n{latest.content}")
    except GraphRecursionError:
        print("  [recursion limit reached, using last answer so far]", flush=True)
        if not raw_parts:
            raw_parts.append("Keine ausreichenden Informationen gefunden.")

    raw_answer = "\n".join(raw_parts)

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
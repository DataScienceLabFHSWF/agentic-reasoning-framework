import os
import sys
from typing import Any

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import after_model, wrap_tool_call
from langchain_ollama import ChatOllama

from agentrf.settings import load_settings
from agentrf.tools import build_chroma_retriever_tool


MODEL_TOOL_CALL_COUNT = 0
EXECUTED_TOOL_CALL_COUNT = 0


@after_model
def log_model_tool_plan(state, runtime) -> dict[str, Any] | None:
    """
    Log only the tool calls the model planned.
    Do not print retrieved chunks or tool results.
    """
    global MODEL_TOOL_CALL_COUNT

    messages = state["messages"]
    last_msg = messages[-1]

    tool_calls = getattr(last_msg, "tool_calls", None) or []
    MODEL_TOOL_CALL_COUNT += len(tool_calls)

    if tool_calls:
        print("\n=== Middleware: model selected tool call(s) ===")
        print(f"count={len(tool_calls)}")
        for i, tc in enumerate(tool_calls, start=1):
            print(f"[{i}] tool={tc.get('name')}")
            print(f"    args={tc.get('args')}")
            print(f"    id={tc.get('id')}")
    
    return None


@wrap_tool_call
def log_tool_execution(request, handler):
    """
    Log only that a tool is about to execute.
    Do not print returned chunks/results.
    """
    global EXECUTED_TOOL_CALL_COUNT

    tc = request.tool_call
    EXECUTED_TOOL_CALL_COUNT += 1

    print(
        "\n=== Middleware: about to execute tool ===\n"
        f"tool={tc.get('name')}\n"
        f"args={tc.get('args')}\n"
        f"id={tc.get('id')}",
        flush=True,
    )

    return handler(request)


def main():
    load_dotenv()
    settings = load_settings("./config/toy.yaml")

    retrieval_tool = build_chroma_retriever_tool(
        chroma_persist_dir=settings.paths.chroma_db_dir,
        model_name=settings.rag.embedding.model,
        top_k=settings.rag.retriever.top_k,
        cache_folder=os.getenv("HF_HOME"),
    )

    llm = ChatOllama(
        model="mistral-small3.2:24b",
        temperature=0.0,
        base_url="http://localhost:11434",
    )

    system_prompt = """You are a helpful research assistant.
Always call the retrieve_information tool before answering questions that require knowledge-base facts.
If the query is complex try to decompose the question into multiple calls to the retrieval tool.
Answer in German.
"""

    agent = create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=system_prompt,
        middleware=[log_model_tool_plan, log_tool_execution],
    )

    test_query = (
        "Welche Reaktor ist KKG? Und welches Unternehmen betreibt das "
        "Kernkraftwerk Grafenrheinfeld? Und in welchem Jahr ging das KKG in "
        "kommerziellen Betrieb? Wie heißt das Standortzwischenlager am KKG? "
        "Wofür steht die Abkürzung RBZ in den KKG-Unterlagen? In welcher "
        "Naturlandschaft liegt der Standort des KRB II und wie breit ist dieser Ort? "
        "Welche Teilsektion der Abbauphase 1 des KKG entspricht dem kernbrennstofffreien Zustand? "
        "Welche Einheit für die biologische Wirkung ionisierender Strahlung auf den Menschen nennt das UVU-Glossar? "
        "Um wie viele Millionen Megawattstunden übertrifft die Gesamtstromproduktion des KRB II die des KKG? "
        "Antwort als JSON mit frage und antwort."
    )

    agent_response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": test_query}
            ]
        }
    )

    print("\n=== Final agent response ===")
    print(agent_response["messages"][-1].content)

    print("\n=== Tool call summary ===")
    print(f"Model-planned tool calls: {MODEL_TOOL_CALL_COUNT}")
    print(f"Executed tool calls:      {EXECUTED_TOOL_CALL_COUNT}")


if __name__ == "__main__":
    main()
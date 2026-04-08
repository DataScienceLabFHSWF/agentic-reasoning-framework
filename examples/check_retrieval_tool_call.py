import os
import sys

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# Chroma workaround:
# https://stackoverflow.com/questions/76958817/streamlit-your-system-has-an-unsupported-version-of-sqlite3-chroma-requires-sq/77199016

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from agentrf.settings import load_settings
from agentrf.tools import build_chroma_retriever_tool


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
    )

    test_query = (
        "Welche Reaktor ist KKG? Und welches Unternehmen betreibt das "
        "Kernkraftwerk Grafenrheinfeld? Und in welchem Jahr ging das KKG in "
        "kommerziellen Betrieb? Wie heißt das Standortzwischenlager am KKG? "
        "Wofür steht die Abkürzung RBZ in den KKG-Unterlagen? In welcher "
        "Naturlandschaft liegt der Standort des KRB II und wie breit ist dieser Ort? "
        "Welche Teilsektion der Abbauphase 1 des KKG entspricht dem kernbrennstofffreien Zustand?"
        "Welche Einheit für die biologische Wirkung ionisierender Strahlung auf den Menschen nennt das UVU‑Glossar?"
        "Um wie viele Millionen Megawattstunden übertrifft die Gesamtstromproduktion des KRB II die des KKG?"
        "Antwort als JSON mit frage und antwort."
    )

    llm_with_tools = llm.bind_tools([retrieval_tool])
    resp = llm_with_tools.invoke(test_query)

    print("=== Raw model response ===")
    print(resp)
    print()
    print("=== Tool calls ===")
    print(resp.tool_calls)

    if resp.tool_calls:
        agent_response = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": test_query}
                ]
            }
        )

        print()
        print("=== Final agent response ===")
        print(agent_response["messages"][-1].content)
        print()
        #print("=== Full agent output ===")
        #print(agent_response)
    else:
        print()
        print("Model did not emit tool calls. The agent will likely not use the retriever tool with this model.")


if __name__ == "__main__":
    main()
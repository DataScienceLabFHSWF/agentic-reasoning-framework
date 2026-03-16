import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import ChatOllama

load_dotenv()

host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "5432")
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "mistral-small3.2:24b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.0,
)

def chatbot(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

with PostgresSaver.from_conn_string(db_uri) as checkpointer:
    # Needed the first time so LangGraph creates its tables
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "demo-thread-1"}}

    result1 = graph.invoke(
        {"messages": [{"role": "user", "content": "Hi, my name is Alice."}]},
        config=config,
    )
    print("Run 1:", result1["messages"][-1].content)

    result2 = graph.invoke(
        {"messages": [{"role": "user", "content": "What do you know about me?"}]},
        config=config,
    )
    print("Run 2:", result2["messages"][-1].content)
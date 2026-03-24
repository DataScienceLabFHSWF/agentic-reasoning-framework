import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
import psycopg2

load_dotenv()

host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "5432")
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

db_uri = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

with psycopg2.connect(db_uri) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print(cur.fetchone())

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "mistral-small3.2:24b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0.0,
)


def chatbot(state: MessagesState):
    print("Chatbot received messages:", state["messages"])
    print("**********************************************")
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

    config = {"configurable": {"thread_id": "demo-thread-2"}}

    result1 = graph.invoke(
        {"messages": [{"role": "user", "content": "Hi, my name is Alice."}]},
        config=config,
    )
    print("Run 1:", result1["messages"][-1].content)

    result2 = graph.invoke(
        {"messages": [{"role": "user", "content": "My name is Bob."}]},
        config=config,
    )
    print("Run 2:", result2["messages"][-1].content)

    result3 = graph.invoke(
        {"messages": [{"role": "user", "content": "List all the facts you know about me."}]},
        config=config,
    )
    print("Run 3:", result3["messages"][-1].content)

import os
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Environment
os.environ["OLLAMA_HOST"] = "http://localhost:11434/v1"

# LLMs
chat_llm = ChatOpenAI(
    model="llama3.1:latest",
    temperature=0,
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

# Tools
@tool
def llm_summarize_10words(txt: str) -> str:
    """Summarize text in exactly 10 words using the LLM."""
    return base_llm.invoke(f"Summarize this text in exactly 10 words: {txt}").content

@tool
def check_10words(text: str) -> bool:
    """Check if a text contains exactly 10 words."""
    return len(text.split()) == 10

tools = [llm_summarize_10words, check_10words]

# Bind tools to chat_llm
chat_llm = chat_llm.bind_tools(tools)

# State schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# LLM node: makes the model decide to call tools
def call_llm(state: State):
    resp = chat_llm.invoke(state["messages"])
    return {"messages": [resp]}

# Tool executor node
tool_node = ToolNode(tools)

# Stop condition: if no more tool_calls, end; otherwise go back to LLM
def decide_next(state: State):
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END

# Build the graph
builder = StateGraph(State)
builder.add_node("llm", call_llm)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", decide_next)
builder.add_edge("tools", "llm")
graph = builder.compile()

# Interactive loop
state = {"messages": []}
while True:
    ui = input("You: ")
    if ui.lower() in {"exit", "quit"}:
        break
    state["messages"].append({"role": "user", "content": ui})
    result = graph.invoke(state)
    state = result  # update state

    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        print("[Tool call was made]")
    else:
        print("[No tool call]")
    print("Agent:", last_msg.content)

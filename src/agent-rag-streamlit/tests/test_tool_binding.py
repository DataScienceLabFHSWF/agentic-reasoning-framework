#!/usr/bin/env python3
import sys
import os
from langchain_ollama import ChatOllama
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent_utils.retriever_tool import RetrieverTool

# Config
CHROMA_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"
PROCESSED_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"
MODEL_NAME = "qwen3:14b"

def test_tool_binding():

    model_name = MODEL_NAME
    print(f"Using model: {model_name}")
    
    # Setup
    retriever = RetrieverTool(CHROMA_DIR, PROCESSED_DIR, k=2)
    tool = retriever.as_langchain_tool()
    llm = ChatOllama(model=model_name, temperature=0.0, base_url="http://localhost:11434")
    llm_with_tools = llm.bind_tools([tool])
    
    # Test query
    query = "Use retrieve_documents tool to find KRB II information"
    print(f"Query: {query}")
    
    # Get response
    response = llm_with_tools.invoke(query)
    
    # Results
    print("*****respone tool call", response.tool_calls)

    tool_calls_made = bool(hasattr(response, 'tool_calls') and response.tool_calls)
    print(f"Tool calls made: {tool_calls_made}")
    print(f"Response: {response.content[:200]}...")
    
    if tool_calls_made:
        print(f"Number of tool calls: {len(response.tool_calls)}")
        for i, call in enumerate(response.tool_calls):
            print(f"  Call {i+1}: {call.get('name', 'unknown')}")

if __name__ == "__main__":
    test_tool_binding()
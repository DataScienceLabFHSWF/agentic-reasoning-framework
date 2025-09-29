#!/usr/bin/env python3
import sys
import os
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agent_utils.retriever_tool import RetrieverTool

# Config
CHROMA_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"
PROCESSED_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"
MODEL = "qwen3:14b"

def test_tool_execution():
    # Get available models
   
    model_name = MODEL
    print(f"Using model: {model_name}")
    
    # Setup
    retriever = RetrieverTool(CHROMA_DIR, PROCESSED_DIR, k=2)
    tools = [retriever.as_langchain_tool()]
    llm = ChatOllama(model=model_name, temperature=0.0, base_url="http://localhost:11434")
    
    # Create agent that actually executes tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the retriever tool when needed to answer the given question in german. If there is not enough context you can run the retriever tool again."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test query
    query = "Using the retrieval tool to answer the following in one word: Wie heißt das Standort Zwischenlager am KKG, das 88 Stellplätze besitzt? "
    print(f"Query: {query}")
    
    # Execute with actual tool calling
    result = agent_executor.invoke({"input": query})
    
    print(f"Final answer: {result['output']}...")

if __name__ == "__main__":
    test_tool_execution()
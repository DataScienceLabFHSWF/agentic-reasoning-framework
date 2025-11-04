#!/usr/bin/env python3
import sys
import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag_utils.retriever import hybrid_retrieve

# Configuration
CHROMA_DIR = "../chroma_db"
PROCESSED_DIR = "../processed_files"
MODEL = "qwen3:32b"
MAX_ITERATIONS = 20

@tool
def search_documents(query: str) -> str:
    """Durchsuche Dokumente mit hybrider Suche.
    
    Falls du nicht findest wonach du suchst, versuche verschiedene Suchbegriffe:
    - Verwende Synonyme oder verwandte Begriffe
    - Suche nach spezifischen Schlüsselwörtern aus der Frage
    - Versuche kürzere, fokussiertere Anfragen
    - Suche sowohl auf Deutsch als auch auf Englisch
    
    Versuche immer mehrere Suchen mit verschiedenen Begriffen bevor du schließt, dass die Information nicht verfügbar ist."""
    
    print(f"🔍 SUCHE: '{query}'")  # Print each search query
    
    docs = hybrid_retrieve(
        query=query,
        chroma_dir=CHROMA_DIR,
        processed_dir=PROCESSED_DIR,
        k=2  # Using k=2 like in original config
    )
    
    if not docs:
        result = "Keine relevanten Dokumente gefunden."
        print(f"❌ ERGEBNIS: {result}")
        return result
    
    results = []
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:400]
        source = doc.metadata.get('filename', 'unknown')
        score = doc.metadata.get('score', 0.0)
        results.append(f"[{i}] {source} (score: {score:.3f})\n{content}")
    
    result = "\n\n".join(results)
    print(f"✅ GEFUNDEN: {len(docs)} Dokumente")
    return result

def test_rag_agent():
    """Test the RAG agent with ReAct loop."""
    model_name = MODEL
    print(f"🤖 Verwende Modell: {model_name}")
    
    # Create LLM with Ollama
    llm = ChatOllama(model=model_name, temperature=0.0, base_url="http://localhost:11434")
    
    # Create ReAct agent with explicit instructions
    system_message = """Du bist ein hartnäckiger Recherche-Assistent. Bei der Suche nach Informationen:

1. Falls deine erste Suche die Antwort nicht findet, versuche verschiedene Suchbegriffe
2. Zerlege komplexe Fragen in kleinere Teile
3. Verwende Synonyme, Schlüsselwörter oder verwandte Begriffe
4. Versuche sowohl deutsche als auch englische Begriffe
5. Gib nicht nach einer Suche auf - versuche mehrere Ansätze
6. Für die Frage nach "Zwischenlager am KKG mit 88 Stellplätzen", versuche Suchen wie:
   - "Zwischenlager KKG 88"
   - "KKG Stellplätze 88" 
   - "Zwischenlager 88 Stellplätze"
   - "KKG interim storage"
   - "88 Stellplätze"
   - "KKG Lager"

Führe immer mehrere Suchversuche mit verschiedenen Begriffen durch, bevor du schließt, dass die Information nicht verfügbar ist."""
    
    print("📋 SYSTEM ANWEISUNGEN: Hartnäckige Suche mit mehreren Versuchen")
    
    agent = create_react_agent(llm, [search_documents], max_iterations=MAX_ITERATIONS, state_modifier=system_message)
    
    # Test query (same as original)
    query = "Using the retrieval tool to answer the following in one word: Welches der beiden Werke, KKG oder KRB II, hat vor der Stilllegung mehr Strom produziert?"
    print(f"❓ FRAGE: {query}")
    print(f"🔄 MAX_ITERATIONEN: {MAX_ITERATIONS}")
    print("=" * 80)
    
    # Execute with ReAct loop
    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    print("=" * 80)
    final_answer = response["messages"][-1].content
    print(f"🎯 FINALE ANTWORT: {final_answer}")
    return final_answer

def ask_question(question: str) -> str:
    """Ask a question using the RAG agent with ReAct loop."""
    llm = ChatOllama(model=MODEL, temperature=0.0, base_url="http://localhost:11434")
    
    system_message = """You are a persistent research assistant. When searching for information:

1. If your first search doesn't find the answer, try different search terms
2. Break down complex questions into smaller parts  
3. Use synonyms, keywords, or related terms
4. Try searching in both German and English
5. Don't give up after one search - try multiple approaches

Always make multiple search attempts with different terms before concluding information is unavailable."""
    
    agent = create_react_agent(llm, [search_documents], max_iterations=MAX_ITERATIONS, state_modifier=system_message)
    response = agent.invoke({"messages": [HumanMessage(content=question)]})
    return response["messages"][-1].content

if __name__ == "__main__":
    test_rag_agent()
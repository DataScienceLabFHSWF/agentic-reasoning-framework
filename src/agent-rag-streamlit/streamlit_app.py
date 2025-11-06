import streamlit as st
import os
import sys
import time
import logging
import yaml  # <-- Import
from typing import Generator, Dict, Any

# --- Start: Load Configuration ---
def load_config():
    """Loads config from YAML file."""
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load config globally
CONFIG = load_config()

# Use config values
paths = CONFIG['paths']
models = CONFIG['models']
settings = CONFIG['settings']

CHROMA_DIR = os.environ.get("CHROMA_DIR", paths['chroma_db'])
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", paths['processed_files'])

INTENT_MODEL = os.environ.get("INTENT_MODEL", models['intent'])
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", models['router'])
REASONING_MODEL = os.environ.get("REASONING_MODEL", models['reasoning'])
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", models['summarizer'])
GENERAL_MODEL = os.environ.get("GENERAL_MODEL", models['general'])
FINAL_ANSWER_MODEL = os.environ.get("FINAL_ANSWER_MODEL", models['final_answer'])

RELEVANCE_THRESHOLD = os.environ.get("RELEVANCE_THRESHOLD", settings['relevance_threshold'])
# --- End: Load Configuration ---


sys.path.append(os.path.dirname(__file__))

from agent_utils.agentic_rag_chat import create_rag_chat

# Page configuration
st.set_page_config(
    page_title="Agentic Reasoning Chat System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
.processing-status {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    background: linear-gradient(90deg, #f0f2f6 0%, #e8eaf6 50%, #f0f2f6 100%);
    border-left: 4px solid #1976d2;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #1976d2;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 10px;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.score-info { 
    background-color: #f5f5f5; 
    padding: 8px; 
    border-radius: 4px; 
    margin: 5px 0;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)



def show_agent_status(agent_type: str, status: str, details: str = ""):
    """Display agent status with professional styling"""
    css_class = f"{agent_type.lower()}-status"
    status_html = f'<div class="agent-status {css_class}"><strong>{agent_type.title()} Agent:</strong> {status}'
    if details:
        status_html += f'<br><small>{details}</small>'
    status_html += '</div>'
    return st.markdown(status_html, unsafe_allow_html=True)

def show_rag_scores(scores: Dict[str, Any]):
    """Display RAG relevance scores and metadata"""
    if scores:
        score_html = '<div class="score-info">'
        score_html += f'<strong>RAG Relevance Analysis:</strong><br>'
        if 'max_score' in scores:
            score_html += f'Maximum Relevance Score: {scores["max_score"]:.3f}<br>'
        if 'threshold' in scores:
            score_html += f'Relevance Threshold: {scores["threshold"]:.3f}<br>'
        if 'docs_count' in scores:
            score_html += f'Documents Retrieved: {scores["docs_count"]}<br>'
        if 'decision' in scores:
            score_html += f'RAG Decision: {scores["decision"]}'
        score_html += '</div>'
        st.markdown(score_html, unsafe_allow_html=True)

def initialize_rag_chat():
    """Initialize RAG chat system with detailed status updates"""
    try:
        status_container = st.container()
        
        with status_container:
            st.info("Initializing Agentic Reasoning Chat System...")
            
            # Model loading status
            model_status = st.empty()
            model_status.markdown("**Loading Models:**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Updated stages to include all agents
            stages = [
                ("Intent Agent", "Loading intent classification model..."),
                ("Router Agent", "Preparing routing intelligence..."),
                ("Reasoning Agent", "Loading reasoning capabilities..."),
                ("Summarizer Agent", "Loading document summarization capabilities..."),
                ("General Agent", "Initializing conversational AI..."),
                ("Final Answer Agent", "Loading final answer generation..."),
                ("Vector Database", "Connecting to ChromaDB..."),
                ("RAG Pipeline", "Assembling retrieval-augmented generation system...")
            ]
            
            for i, (component, description) in enumerate(stages):
                status_text.text(f"Loading {component}: {description}")
                progress_bar.progress((i + 1) / len(stages))
                time.sleep(0.5)  # Simulate loading time
            
            # Create the actual RAG chat with all models
            rag_chat = create_rag_chat(
                chroma_dir=CHROMA_DIR,
                processed_dir=PROCESSED_DIR,
                intent_model=INTENT_MODEL,
                router_model=ROUTER_MODEL,
                reasoning_model=REASONING_MODEL,
                summarizer_model=SUMMARIZER_MODEL,
                general_model=GENERAL_MODEL,
                final_answer_model=FINAL_ANSWER_MODEL,
                relevance_threshold=RELEVANCE_THRESHOLD
            )
            
            status_text.text("Agentic RAG system ready!")
            time.sleep(0.5)
            
            # Clear initialization UI
            status_container.empty()
            
        return rag_chat
        
    except Exception as e:
        st.error(f"Failed to initialize Agentic RAG Chat System: {str(e)}")
        return None

def show_processing_status(message: str):
    """Display single processing status with spinner"""
    status_html = f'''
    <div class="processing-status">
        <div class="spinner"></div>
        <strong>{message}</strong>
    </div>
    '''
    return st.markdown(status_html, unsafe_allow_html=True)

def stream_response(rag_chat, prompt: str) -> Generator[str, None, None]:
    """Stream response with real-time agent activity updates"""
    
    # Create single status container
    status_container = st.empty()
    
    try:
        # Start the actual chat process and monitor it
        with status_container.container():
            show_processing_status("Router Agent: Analyzing query intent and determining processing strategy...")
        
        # Call the actual RAG chat method
        response = rag_chat.chat(prompt)
        
        # Clear status container before streaming response
        status_container.empty()
        
        # Stream the response word by word
        words = response.split()
        streamed_text = ""
        
        for word in words:
            streamed_text += word + " "
            yield streamed_text
            time.sleep(0.05)  # Adjust streaming speed
            
    except Exception as e:
        status_container.empty()
        yield f"Error generating response: {str(e)}"

def get_system_info(rag_chat):
    """Get dynamic system information from the RAG chat instance"""
    if not rag_chat:
        return {
            "agents": ["System not initialized"],
            "threshold": "N/A",
            "status": "error"
        }
    
    try:
        # Extract actual configuration from the rag_chat object
        agents = []
        
        # Get all agent information from the rag_chat instance
        if hasattr(rag_chat, 'intent_agent'):
            agents.append(f"Intent Agent ({INTENT_MODEL})")
        
        if hasattr(rag_chat, 'router_agent'):
            agents.append(f"Router Agent ({ROUTER_MODEL})")
            
        if hasattr(rag_chat, 'retriever_agent'):
            agents.append("Retrieval Agent (ChromaDB + Hybrid Search)")
            
        if hasattr(rag_chat, 'reasoning_agent'):
            agents.append(f"Reasoning Agent ({REASONING_MODEL})")
        
        if hasattr(rag_chat, 'summarizer_agent'):
            agents.append(f"Summarizer Agent ({SUMMARIZER_MODEL})")
        
        if hasattr(rag_chat, 'general_agent'):
            agents.append(f"General Agent ({GENERAL_MODEL})")
            
        if hasattr(rag_chat, 'final_answer_agent'):
            agents.append(f"Final Answer Agent ({FINAL_ANSWER_MODEL})")
        
        # Get threshold
        threshold = getattr(rag_chat, 'relevance_threshold', RELEVANCE_THRESHOLD)
        
        return {
            "agents": agents if agents else ["Configuration not accessible"],
            "threshold": threshold,
            "status": "operational"
        }
        
    except Exception as e:
        return {
            "agents": [f"Error reading configuration: {str(e)}"],
            "threshold": "N/A",
            "status": "error"
        }

def main():
    st.title("Agentic Reasoning Chat System")
    st.markdown("**Intelligent Document Retrieval and Conversational AI**")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = initialize_rag_chat()
    
    # Sidebar
    with st.sidebar:
        st.header("System Controls")
        
        if st.button("Clear Conversation", type="primary"):
            st.session_state.messages = []
            if st.session_state.rag_chat:
                st.session_state.rag_chat.clear_chat_history()
            st.rerun()
        
        st.markdown("---")
        st.subheader("System Information")
        
        # Get dynamic system info
        system_info = get_system_info(st.session_state.get("rag_chat"))
        
        st.markdown("**Active Agents:**")
        for agent in system_info["agents"]:
            st.markdown(f"- {agent}")
        
        st.markdown(f"**Relevance Threshold:** {system_info['threshold']}")
        
        # System status indicator
        if system_info["status"] == "operational":
            st.success("✅ All agents operational")
        else:
            st.error("❌ System initialization required")
        
        # Additional workflow info
        st.markdown("---")
        st.subheader("Workflow Overview")
        st.markdown("""
        **Processing Flow:**
        1. **Intent Classification** - Determines query type
        2. **Document Retrieval** - Finds relevant content
        3. **Relevance Routing** - Decides processing path
        4. **Reasoning** - Analyzes documents and context
        5. **Summarization** - Creates user-friendly response
        6. **Final Answer** - Provides concise output
        """)
    
    # Check if RAG system is ready
    if not st.session_state.rag_chat:
        st.error("Agentic RAG system failed to initialize. Please check your configuration and restart.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and stream response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                full_response = ""
                for partial_response in stream_response(st.session_state.rag_chat, prompt):
                    response_placeholder.write(partial_response)
                    full_response = partial_response
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": full_response.strip()})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
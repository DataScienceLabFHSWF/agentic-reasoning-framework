import streamlit as st
import os
import sys
import time
import logging
from typing import Generator, Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

from agent_utils.agentic_rag_chat import create_rag_chat

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Chat System",
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

# Constants
CHROMA_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"
PROCESSED_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"
ROUTER_MODEL = "qwen3:14b"
SUMMARIZER_MODEL = "mistral:latest"
GENERAL_MODEL = "mistral:latest"
RELEVANCE_THRESHOLD = 0.15

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
            st.info("Initializing Agentic RAG Chat System...")
            
            # Model loading status
            model_status = st.empty()
            model_status.markdown("**Loading Models:**")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate model loading stages
            stages = [
                ("Router Agent (qwen3:14b)", "Preparing routing intelligence..."),
                ("Summarizer Agent (mistral:latest)", "Loading document summarization capabilities..."),
                ("General Agent (qwen3:14b)", "Initializing conversational AI..."),
                ("Vector Database", "Connecting to ChromaDB..."),
                ("RAG Pipeline", "Assembling retrieval-augmented generation system...")
            ]
            
            for i, (component, description) in enumerate(stages):
                status_text.text(f"Loading {component}: {description}")
                progress_bar.progress((i + 1) / len(stages))
                time.sleep(0.5)  # Simulate loading time
            
            # Create the actual RAG chat
            rag_chat = create_rag_chat(
                chroma_dir=CHROMA_DIR,
                processed_dir=PROCESSED_DIR,
                router_model=ROUTER_MODEL,
                summarizer_model=SUMMARIZER_MODEL,
                general_model=GENERAL_MODEL,
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
        
        # Get router model if available
        if hasattr(rag_chat, 'router') and hasattr(rag_chat.router, 'model'):
            agents.append(f"Router Agent ({rag_chat.router.model})")
        elif hasattr(rag_chat, 'router_model'):
            agents.append(f"Router Agent ({rag_chat.router_model})")
        
        # Get retrieval/vector database info
        if hasattr(rag_chat, 'vector_store') or hasattr(rag_chat, 'chroma_client'):
            agents.append("Retrieval Agent (ChromaDB)")
        
        # Get summarizer model if available
        if hasattr(rag_chat, 'summarizer') and hasattr(rag_chat.summarizer, 'model'):
            agents.append(f"Summarizer Agent ({rag_chat.summarizer.model})")
        elif hasattr(rag_chat, 'summarizer_model'):
            agents.append(f"Summarizer Agent ({rag_chat.summarizer_model})")
        
        # Get general model if available
        if hasattr(rag_chat, 'general_agent') and hasattr(rag_chat.general_agent, 'model'):
            agents.append(f"General Agent ({rag_chat.general_agent.model})")
        elif hasattr(rag_chat, 'general_model'):
            agents.append(f"General Agent ({rag_chat.general_model})")
        
        # Get threshold
        threshold = getattr(rag_chat, 'relevance_threshold', 0.3)
        
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
    st.title("Agentic RAG Chat System")
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
        st.markdown(f"""
        **Active Agents:**
        - Router Agent ({ROUTER_MODEL})
        - Retrieval Agent (ChromaDB)
        - Summarizer Agent ({SUMMARIZER_MODEL})
        - General Agent ({GENERAL_MODEL})
        
        **Relevance Threshold:** {RELEVANCE_THRESHOLD}
        """)
        
        if st.session_state.get("rag_chat"):
            st.success("All agents operational")
        else:
            st.error("System initialization required")
    
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
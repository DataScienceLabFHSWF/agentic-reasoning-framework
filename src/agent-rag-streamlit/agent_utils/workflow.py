"""
workflow.py
LangGraph workflow definition for RAG chat
"""

import logging
from langgraph.graph import StateGraph, END

from .chat_state import ChatState
from .router_agent import RouterAgent
from .retriever_agent import RetrieverAgent
from .summarizer_agent import SummarizerAgent
from .general_agent import GeneralAgent

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """LangGraph workflow for RAG-based chat"""
    
    def __init__(
        self,
        router_agent: RouterAgent,
        retriever_agent: RetrieverAgent,
        summarizer_agent: SummarizerAgent,
        general_agent: GeneralAgent
    ):
        self.router_agent = router_agent
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent
        self.general_agent = general_agent
        
        logger.info("Building RAG workflow")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        logger.info("RAG workflow compiled successfully")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ChatState)
        
        # Add nodes - note the new flow: retrieve first, then route based on relevance
        workflow.add_node("rag_retriever", self.retriever_agent.retrieve_documents)
        workflow.add_node("router", self.router_agent.route_query)
        workflow.add_node("summarizer", self.summarizer_agent.summarize_response)
        workflow.add_node("general_response", self.general_agent.general_response)
        
        # Define the flow: retrieve documents first, then route based on relevance scores
        workflow.set_entry_point("rag_retriever")
        
        # After retrieval, always go to router to evaluate relevance
        workflow.add_edge("rag_retriever", "router")
        
        # Conditional routing based on document relevance scores
        workflow.add_conditional_edges(
            "router",
            self.router_agent.route_decision,
            {
                "relevant": "summarizer",
                "not_relevant": "general_response"
            }
        )
        
        # End points
        workflow.add_edge("summarizer", END)
        workflow.add_edge("general_response", END)
        
        logger.info("Workflow graph constructed: retrieve -> route -> [summarize|general]")
        return workflow
    
    def invoke(self, initial_state: ChatState) -> ChatState:
        """Execute the workflow with the given initial state"""
        logger.info("Starting workflow execution")
        result = self.app.invoke(initial_state)
        logger.info("Workflow execution completed")
        return result
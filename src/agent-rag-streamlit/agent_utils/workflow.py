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
from .intent_agent import IntentClassificationAgent

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """LangGraph workflow for RAG-based chat with intent classification"""
    
    def __init__(
        self,
        intent_agent: IntentClassificationAgent,
        router_agent: RouterAgent,
        retriever_agent: RetrieverAgent,
        summarizer_agent: SummarizerAgent,
        general_agent: GeneralAgent
    ):
        self.intent_agent = intent_agent
        self.router_agent = router_agent
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent
        self.general_agent = general_agent
        
        logger.info("Building RAG workflow with intent classification")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        logger.info("RAG workflow compiled successfully")
        
        # Generate workflow diagram
        self._generate_workflow_diagram()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow: intent -> [retrieve -> route -> [summarize|general]] | general"""
        workflow = StateGraph(ChatState)
        
        # Add nodes - new flow starts with intent classification
        workflow.add_node("intent_classifier", self.intent_agent.classify_intent)
        workflow.add_node("rag_retriever", self.retriever_agent.retrieve_documents)
        workflow.add_node("router", self.router_agent.route_query)
        workflow.add_node("summarizer", self.summarizer_agent.summarize_response)
        workflow.add_node("general_response", self.general_agent.general_response)
        
        # Define the flow: classify intent first
        workflow.set_entry_point("intent_classifier")
        
        # Route based on intent classification
        workflow.add_conditional_edges(
            "intent_classifier",
            self.intent_agent.route_decision,
            {
                "corpus_relevant": "rag_retriever",
                "general": "general_response"
            }
        )
        
        # RAG flow: retrieve -> route -> [summarize|general]
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
        
        logger.info("Workflow graph constructed: intent -> [retrieve -> route -> [summarize|general]] | general")
        return workflow
    
    def _generate_workflow_diagram(self):
        """Generate and save workflow diagram automatically"""
        try:
            graph = self.app.get_graph()
            png_bytes = graph.draw_mermaid_png()
            out_path = "current_workflow.png"
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            logger.info(f"Workflow diagram saved to {out_path}")
        except Exception as e:
            logger.warning(f"Failed to generate workflow diagram: {e}")
    
    def invoke(self, initial_state: ChatState) -> ChatState:
        """Execute the workflow with the given initial state"""
        logger.info("Starting workflow execution")
        result = self.app.invoke(initial_state)
        logger.info("Workflow execution completed")
        return result
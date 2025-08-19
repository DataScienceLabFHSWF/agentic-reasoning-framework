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
from .follow_up_agent import FollowUpAgent
from .human_intervention_agent import HumanInterventionAgent

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """LangGraph workflow for RAG-based chat with intent classification, follow-ups, and human intervention"""
    
    def __init__(
        self,
        intent_agent: IntentClassificationAgent,
        router_agent: RouterAgent,
        retriever_agent: RetrieverAgent,
        summarizer_agent: SummarizerAgent,
        general_agent: GeneralAgent,
        follow_up_agent: FollowUpAgent,
        human_intervention_agent: HumanInterventionAgent
    ):
        self.intent_agent = intent_agent
        self.router_agent = router_agent
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent
        self.general_agent = general_agent
        self.follow_up_agent = follow_up_agent
        self.human_intervention_agent = human_intervention_agent
        
        logger.info("Building enhanced RAG workflow with follow-ups and human intervention")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        logger.info("Enhanced RAG workflow compiled successfully")
        
        # Generate workflow diagram
        self._generate_workflow_diagram()
    
    def _build_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow with multiple routing paths"""
        workflow = StateGraph(ChatState)
        
        # Add all nodes
        workflow.add_node("intent_classifier", self.intent_agent.classify_intent)
        workflow.add_node("human_intervention", self.human_intervention_agent.handle_intervention)
        workflow.add_node("follow_up", self.follow_up_agent.handle_follow_up)
        workflow.add_node("rag_retriever", self.retriever_agent.retrieve_documents)
        workflow.add_node("router", self.router_agent.route_query)
        workflow.add_node("summarizer", self.summarizer_agent.summarize_response)
        workflow.add_node("general_response", self.general_agent.general_response)
        
        # Entry point
        workflow.set_entry_point("intent_classifier")
        
        # Route based on intent classification (now includes follow-up and human intervention)
        workflow.add_conditional_edges(
            "intent_classifier",
            self.intent_agent.route_decision,
            {
                "human_intervention": "human_intervention",
                "follow_up": "follow_up",
                "corpus_relevant": "rag_retriever",
                "general": "general_response"
            }
        )
        
        # Human intervention routing
        workflow.add_conditional_edges(
            "human_intervention",
            self.human_intervention_agent.route_decision,
            {
                "force_rag": "rag_retriever",
                "use_context": "follow_up",
                "general": "general_response"
            }
        )
        
        # Standard RAG flow
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
        workflow.add_edge("follow_up", END)
        workflow.add_edge("summarizer", END)
        workflow.add_edge("general_response", END)
        
        logger.info("Enhanced workflow: intent -> [human_intervention|follow_up|retrieve -> route -> [summarize|general]] | general")
        return workflow
    
    def _generate_workflow_diagram(self):
        """Generate and save workflow diagram automatically"""
        try:
            graph = self.app.get_graph()
            png_bytes = graph.draw_mermaid_png()
            out_path = "enhanced_workflow.png"
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            logger.info(f"Enhanced workflow diagram saved to {out_path}")
        except Exception as e:
            logger.warning(f"Failed to generate workflow diagram: {e}")
    
    def invoke(self, initial_state: ChatState) -> ChatState:
        """Execute the workflow with the given initial state"""
        logger.info("Starting enhanced workflow execution")
        result = self.app.invoke(initial_state)
        logger.info("Enhanced workflow execution completed")
        return result
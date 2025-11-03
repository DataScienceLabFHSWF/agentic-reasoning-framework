#!/usr/bin/env python3
"""
plot_dummy_rag_workflow.py
Self-contained dummy LangGraph workflow + plotter for a RAG chat flow.

Flow (matches your logic):
retrieve (rag_retriever) -> router -> { summarizer | general_response } -> END

Requires:
- langgraph
- graphviz (for Mermaid PNG export)
"""

import logging
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("dummy_plotter")

# --- Dummy ChatState (minimal fields for this flow) ---
class ChatState(TypedDict):
    """State of the conversation for the RAG workflow."""
    query: str
    documents: Optional[List[str]]
    relevance_scores: Optional[List[float]]
    response: Optional[str]

# --- Dummy agent base ---
class DummyAgent:
    def __init__(self, name: str):
        self.name = name

# --- Dummy RetrieverAgent ---
class RetrieverAgent(DummyAgent):
    def retrieve_documents(self, state: ChatState) -> ChatState:
        logger.info("DummyRetriever: retrieving documents")
        # Dummy docs + pretend scores
        state["documents"] = ["doc A", "doc B", "doc C"]
        state["relevance_scores"] = [0.82, 0.41, 0.15]
        return state

# --- Dummy RouterAgent ---
class RouterAgent(DummyAgent):
    def route_query(self, state: ChatState) -> ChatState:
        logger.info("DummyRouter: evaluating relevance for routing")
        # No mutation needed; decision is made in route_decision()
        return state

    def route_decision(self, state: ChatState) -> str:
        # Simple dummy rule: if any score >= 0.7 -> "relevant" else "not_relevant"
        scores = state.get("relevance_scores") or []
        decision = "relevant" if any(s >= 0.7 for s in scores) else "not_relevant"
        logger.info(f"DummyRouter: route_decision -> {decision}")
        return decision

# --- Dummy SummarizerAgent ---
class SummarizerAgent(DummyAgent):
    def summarize_response(self, state: ChatState) -> ChatState:
        logger.info("DummySummarizer: summarizing retrieved content")
        state["response"] = "dummy_summary_answer_based_on_retrieved_docs"
        return state

# --- Dummy GeneralAgent (fallback) ---
class GeneralAgent(DummyAgent):
    def general_response(self, state: ChatState) -> ChatState:
        logger.info("DummyGeneral: producing general fallback answer")
        state["response"] = "dummy_general_answer"
        return state

# --- Dummy RAGWorkflow matching your structure ---
class RAGWorkflow:
    """LangGraph workflow for RAG-based chat (dummy implementation)."""

    def __init__(
        self,
        router_agent: RouterAgent,
        retriever_agent: RetrieverAgent,
        summarizer_agent: SummarizerAgent,
        general_agent: GeneralAgent,
    ):
        self.router_agent = router_agent
        self.retriever_agent = retriever_agent
        self.summarizer_agent = summarizer_agent
        self.general_agent = general_agent

        logger.info("Building dummy RAG workflow")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        logger.info("Dummy RAG workflow compiled successfully")

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow: retrieve -> route -> [summarize|general]"""
        workflow = StateGraph(ChatState)

        # Nodes
        workflow.add_node("rag_retriever", self.retriever_agent.retrieve_documents)
        workflow.add_node("router", self.router_agent.route_query)
        workflow.add_node("summarizer", self.summarizer_agent.summarize_response)
        workflow.add_node("general_response", self.general_agent.general_response)

        # Entry
        workflow.set_entry_point("rag_retriever")

        # Edges
        workflow.add_edge("rag_retriever", "router")

        # Conditional edges based on dummy route decision
        workflow.add_conditional_edges(
            "router",
            self.router_agent.route_decision,
            {
                "relevant": "summarizer",
                "not_relevant": "general_response",
            },
        )

        # Terminal edges
        workflow.add_edge("summarizer", END)
        workflow.add_edge("general_response", END)

        logger.info("Workflow graph constructed: retrieve -> route -> [summarize|general]")
        return workflow

    def invoke(self, initial_state: ChatState) -> ChatState:
        logger.info("Starting dummy workflow execution")
        result = self.app.invoke(initial_state)
        logger.info("Dummy workflow execution completed")
        return result


if __name__ == "__main__":
    # Instantiate dummy agents
    router = RouterAgent("router")
    retriever = RetrieverAgent("retriever")
    summarizer = SummarizerAgent("summarizer")
    general = GeneralAgent("general")

    # Build workflow
    workflow = RAGWorkflow(
        router_agent=router,
        retriever_agent=retriever,
        summarizer_agent=summarizer,
        general_agent=general,
    )

    # Plot graph to PNG using Mermaid exporter
    graph = workflow.app.get_graph()
    png_bytes = graph.draw_mermaid_png()

    out_path = "rag_workflow.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)

    print(f"Workflow diagram saved to {out_path}")

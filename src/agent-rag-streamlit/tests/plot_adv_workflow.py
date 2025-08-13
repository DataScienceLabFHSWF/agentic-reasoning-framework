import logging
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Define the state
class ChatState(TypedDict):
    """State of the conversation."""
    query: str
    documents: Optional[List[Document]]
    rag_answer: Optional[str]
    fine_tuned_answer: Optional[str]
    combined_answer: Optional[str]
    route_option: Optional[str]
    response: Optional[str]

# Dummy agent base class
class DummyAgent:
    def __init__(self, name: str):
        self.name = name

# Query rewriter
class QueryRewriterAgent(DummyAgent):
    def rewrite_query(self, state: ChatState) -> ChatState:
        logger.info("Rewriting query for better retrieval")
        # Placeholder: copy query unchanged
        state["route_option"] = None
        return state

# Router decides path: rag, fine, both or general
class RouterAgent(DummyAgent):
    def route_query(self, state: ChatState) -> ChatState:
        logger.info("Routing based on rewritten query")
        return state

    def route_decision(self, state: ChatState) -> str:
        # Hard‑coded decision for demonstration
        return "both"  # could be "rag", "fine_tuned", "general"

# RAG generator
class RAGAgent(DummyAgent):
    def rag_generate(self, state: ChatState) -> ChatState:
        logger.info("Retrieving docs and drafting answer")
        state["rag_answer"] = "rag_answer"
        return state

# Fine‑tuned model
class FineTunedAgent(DummyAgent):
    def fine_generate(self, state: ChatState) -> ChatState:
        logger.info("Generating answer with fine‑tuned model")
        state["fine_tuned_answer"] = "fine_tuned_answer"
        return state

# Combine answers
class AnswerCombinerAgent(DummyAgent):
    def combine_answers(self, state: ChatState) -> ChatState:
        logger.info("Combining RAG and fine‑tuned answers")
        rag = state.get("rag_answer") or ""
        fine = state.get("fine_tuned_answer") or ""
        state["combined_answer"] = f"{rag} + {fine}"
        return state

# Answer checker / summarizer
class AnswerCheckerAgent(DummyAgent):
    def check_answer(self, state: ChatState) -> ChatState:
        logger.info("Checking final answer for quality")
        state["response"] = state.get("combined_answer") or state.get("rag_answer") or state.get("fine_tuned_answer")
        return state

# General/fallback answer
class GeneralAgent(DummyAgent):
    def general_response(self, state: ChatState) -> ChatState:
        logger.info("Providing general answer")
        state["response"] = "general_answer"
        return state

# Build the workflow
class SophisticatedRAGWorkflow:
    def __init__(
        self,
        query_rewriter: QueryRewriterAgent,
        router: RouterAgent,
        rag_agent: RAGAgent,
        fine_agent: FineTunedAgent,
        combiner: AnswerCombinerAgent,
        checker: AnswerCheckerAgent,
        general_agent: GeneralAgent
    ):
        self.query_rewriter = query_rewriter
        self.router = router
        self.rag_agent = rag_agent
        self.fine_agent = fine_agent
        self.combiner = combiner
        self.checker = checker
        self.general_agent = general_agent

        logger.info("Building sophisticated agentic RAG workflow")
        workflow = self._build_workflow()
        self.app = workflow.compile()
        logger.info("Workflow compiled successfully")

    def _build_workflow(self) -> StateGraph:
        g = StateGraph(ChatState)

        # Add nodes
        g.add_node("query_rewriter", self.query_rewriter.rewrite_query)
        g.add_node("router", self.router.route_query)
        g.add_node("rag_generator", self.rag_agent.rag_generate)
        g.add_node("fine_tuned_model", self.fine_agent.fine_generate)
        g.add_node("combine_answers", self.combiner.combine_answers)
        g.add_node("answer_checker", self.checker.check_answer)
        g.add_node("general_response", self.general_agent.general_response)

        # Entry point
        g.set_entry_point("query_rewriter")
        g.add_edge("query_rewriter", "router")

        # Router decisions: choose which path to take
        g.add_conditional_edges(
            "router",
            self.router.route_decision,
            {
                "rag": "rag_generator",
                "fine_tuned": "fine_tuned_model",
                "both": "rag_generator",
                "general": "general_response",
            },
        )

        # For the RAG path
        g.add_edge("rag_generator", "combine_answers")
        # For the fine‑tuned path
        g.add_edge("fine_tuned_model", "combine_answers")
        # For the combined path: call both rag and fine‑tuned
        g.add_edge("rag_generator", "fine_tuned_model")

        # Combine answers → check → end
        g.add_edge("combine_answers", "answer_checker")
        g.add_edge("answer_checker", END)

        # General response → end
        g.add_edge("general_response", END)

        return g

    def invoke(self, state: ChatState) -> ChatState:
        return self.app.invoke(state)

# Example: instantiate dummy agents and workflow
query_rewriter = QueryRewriterAgent("rewriter")
router = RouterAgent("router")
rag_agent = RAGAgent("rag_agent")
fine_agent = FineTunedAgent("fine_agent")
combiner = AnswerCombinerAgent("combiner")
checker = AnswerCheckerAgent("checker")
general = GeneralAgent("general")

workflow = SophisticatedRAGWorkflow(
    query_rewriter,
    router,
    rag_agent,
    fine_agent,
    combiner,
    checker,
    general
)

# Build the graph image (requires Graphviz and LangGraph installed)
graph = workflow.app.get_graph()
png_bytes = graph.draw_mermaid_png()
with open("adv_agentic_rag_workflow.png", "wb") as f:
    f.write(png_bytes)
print("Workflow diagram saved to agentic_rag_workflow.png")

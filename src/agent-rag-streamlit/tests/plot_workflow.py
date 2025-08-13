import logging
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from typing import TypedDict, Optional, List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ===== State =====
class ChatState(TypedDict):
    """Represents the state of the chat."""
    query: str
    documents: Optional[List[Document]]
    response: Optional[str]
    relevance_scores: Optional[List[float]]

# ===== Dummy Agents =====
class DummyAgent:
    def __init__(self, name: str):
        self.name = name

    def execute(self, state: ChatState) -> ChatState:
        logger.info(f"Executing dummy node: {self.name}")
        return state

class MiniRetrieverAgent(DummyAgent):
    """'Mini-RAG' lightweight retrieval (e.g., BM25 small / cached index)."""
    def mini_retrieve(self, state: ChatState) -> ChatState:
        logger.info("Mini-RAG: lightweight retrieval for quick relevance probe.")
        # no-op: keep state as is; in real code populate relevance_scores
        return state

class RouterAgent(DummyAgent):
    """Routes based on mini-RAG relevance outcomes."""
    def route_query(self, state: ChatState) -> ChatState:
        logger.info("Router: assessing mini-RAG relevance.")
        return state

    def route_decision(self, state: ChatState) -> str:
        # Hardcoded for plotting; swap with real relevance logic.
        return "relevant"  # or "not_relevant"

class FullRAGAgent(DummyAgent):
    """Full-RAG pipeline: heavy retrieval + answer drafting."""
    def full_rag(self, state: ChatState) -> ChatState:
        logger.info("Full-RAG: deep retrieval + draft answer generation.")
        return state

class AnswerCheckerAgent(DummyAgent):
    """Checks/cleans the drafted answer (summarizer/validator)."""
    def check_answer(self, state: ChatState) -> ChatState:
        logger.info("AnswerCheck: verifying and tightening the answer.")
        return state

class GeneralAgent(DummyAgent):
    """General fallback when query isn't about the corpus."""
    def general_response(self, state: ChatState) -> ChatState:
        logger.info("General: answering without RAG.")
        return state

# ===== Workflow =====
class RAGWorkflow:
    """LangGraph workflow for Agentic RAG with Mini-RAG gate."""

    def __init__(
        self,
        mini_agent: MiniRetrieverAgent,
        router_agent: RouterAgent,
        full_rag_agent: FullRAGAgent,
        answer_checker: AnswerCheckerAgent,
        general_agent: GeneralAgent
    ):
        self.mini_agent = mini_agent
        self.router_agent = router_agent
        self.full_rag_agent = full_rag_agent
        self.answer_checker = answer_checker
        self.general_agent = general_agent

        logger.info("Building Agentic RAG workflow (Mini-RAG → Router → Full-RAG/General → Check)")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        logger.info("Workflow compiled successfully")

    def _build_workflow(self) -> StateGraph:
        g = StateGraph(ChatState)

        # Nodes
        g.add_node("mini_rag", self.mini_agent.mini_retrieve)            # Mini-RAG (cheap)
        g.add_node("router", self.router_agent.route_query)              # Router over mini relevance
        g.add_node("full_rag", self.full_rag_agent.full_rag)            # Full-RAG (heavy)
        g.add_node("answer_check", self.answer_checker.check_answer)     # Summarizer/Verifier
        g.add_node("general_response", self.general_agent.general_response)

        # Flow
        g.set_entry_point("mini_rag")
        g.add_edge("mini_rag", "router")

        # Conditional split: relevant → Full-RAG → Check; else → General
        g.add_conditional_edges(
            "router",
            self.router_agent.route_decision,
            {
                "relevant": "full_rag",
                "not_relevant": "general_response",
            },
        )

        # Endings
        g.add_edge("full_rag", "answer_check")
        g.add_edge("answer_check", END)
        g.add_edge("general_response", END)

        logger.info("Graph: mini_rag → router → [full_rag → answer_check | general_response]")
        return g

    def invoke(self, initial_state: ChatState) -> ChatState:
        logger.info("Starting workflow execution")
        out = self.app.invoke(initial_state)
        logger.info("Workflow execution completed")
        return out

# ===== Instantiate & Render =====
mini_agent = MiniRetrieverAgent(name="mini_rag")
router_agent = RouterAgent(name="router")
full_rag_agent = FullRAGAgent(name="full_rag")
answer_checker = AnswerCheckerAgent(name="answer_check")
general_agent = GeneralAgent(name="general")

rag_workflow = RAGWorkflow(
    mini_agent=mini_agent,
    router_agent=router_agent,
    full_rag_agent=full_rag_agent,
    answer_checker=answer_checker,
    general_agent=general_agent
)

compiled_app = rag_workflow.app
graph = compiled_app.get_graph()
png_bytes = graph.draw_mermaid_png()

output_file = "agentic_rag_workflow.png"
with open(output_file, "wb") as f:
    f.write(png_bytes)

print(f"Workflow diagram saved to {output_file}")

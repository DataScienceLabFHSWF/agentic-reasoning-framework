"""
workflow.py
LangGraph workflow definition for RAG chat
"""

import logging
import os
from langgraph.graph import StateGraph, END

from .chat_state import ChatState
from .router_agent import RouterAgent
from .retriever_agent import RetrieverAgent
from .reasoning_agent import ReasoningAgent  # New import
from .summarizer_agent import SummarizerAgent
from .general_agent import GeneralAgent
from .intent_agent import IntentClassificationAgent
from dotenv import load_dotenv
from .final_answer_agent import FinalAnswerAgent

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """LangGraph workflow for RAG-based chat with intent classification and reasoning"""
    
    def __init__(
        self,
        intent_agent: IntentClassificationAgent,
        router_agent: RouterAgent,
        retriever_agent: RetrieverAgent,
        reasoning_agent: ReasoningAgent,  # New agent parameter
        summarizer_agent: SummarizerAgent,
        general_agent: GeneralAgent,
        final_answer_agent: FinalAnswerAgent
    ):
        self.intent_agent = intent_agent
        self.router_agent = router_agent
        self.retriever_agent = retriever_agent
        self.reasoning_agent = reasoning_agent  # Assign the new agent
        self.summarizer_agent = summarizer_agent
        self.general_agent = general_agent
        self.final_answer_agent = final_answer_agent
        
        # Check for the final_eval environment variable
        load_dotenv()  # Load environment variables from .env file
        self.final_eval = os.environ.get("final_eval", "succinct")
        
        logger.info("Building RAG workflow with intent classification and reasoning")
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
        logger.info("RAG workflow compiled successfully")
        
        # Generate workflow diagram
        self._generate_workflow_diagram()
    
    def _get_agent_tools(self):
        """Extract tool information from agents"""
        agent_tools = {}
        
        # Check each agent for available tools
        for agent_name, agent in [
            ("intent_classifier", self.intent_agent),
            ("rag_retriever", self.retriever_agent),
            ("router", self.router_agent),
            ("reasoning", self.reasoning_agent),
            ("summarizer", self.summarizer_agent),
            ("general_response", self.general_agent),
            ("final_answer", self.final_answer_agent)
        ]:
            tools = []
            
            # Try to get tools from different possible attributes
            if hasattr(agent, 'tools') and agent.tools:
                tools = [tool.name if hasattr(tool, 'name') else str(tool) for tool in agent.tools]
            elif hasattr(agent, 'tool_names') and agent.tool_names:
                tools = agent.tool_names
            elif hasattr(agent, '_tools') and agent._tools:
                tools = [tool.name if hasattr(tool, 'name') else str(tool) for tool in agent._tools]
            elif hasattr(agent, 'llm') and hasattr(agent.llm, 'tools') and agent.llm.tools:
                tools = [tool.name if hasattr(tool, 'name') else str(tool) for tool in agent.llm.tools]
            
            # Special handling for reasoning agent with retriever_tool
            if agent_name == "reasoning" and hasattr(agent, 'retriever_tool'):
                tools.append("retriever_tool")
                logger.info(f"Found retriever_tool in reasoning agent")
            
            # Check for llm_with_tools (bound tools)
            if hasattr(agent, 'llm_with_tools') and agent.llm_with_tools:
                # Try to extract tool info from bound LLM
                if hasattr(agent.llm_with_tools, 'bound_tools'):
                    bound_tools = [tool.name if hasattr(tool, 'name') else str(tool) for tool in agent.llm_with_tools.bound_tools]
                    tools.extend(bound_tools)
                elif hasattr(agent.llm_with_tools, 'kwargs') and 'tools' in agent.llm_with_tools.kwargs:
                    bound_tools = [tool.name if hasattr(tool, 'name') else str(tool) for tool in agent.llm_with_tools.kwargs['tools']]
                    tools.extend(bound_tools)
                else:
                    # If we can't extract specific tool names but know tools are bound
                    tools.append("bound_tools")
            
            # Remove duplicates while preserving order
            tools = list(dict.fromkeys(tools))
            
            if tools:
                agent_tools[agent_name] = tools
                logger.info(f"Found tools for {agent_name}: {tools}")
        
        return agent_tools
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow: intent -> [retrieve -> route -> [reasoning -> summarize|general]] | general"""
        workflow = StateGraph(ChatState)
        
        # Add nodes - new flow includes reasoning agent
        workflow.add_node("intent_classifier", self.intent_agent.classify_intent)
        workflow.add_node("rag_retriever", self.retriever_agent.retrieve_documents)
        workflow.add_node("router", self.router_agent.route_query)
        workflow.add_node("reasoning", self.reasoning_agent.reason_over_documents)  # New node
        workflow.add_node("summarizer", self.summarizer_agent.summarize_response)
        workflow.add_node("general_response", self.general_agent.general_response)
        workflow.add_node("final_answer", self.final_answer_agent.get_final_answer)
        
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
        
        # RAG flow: retrieve -> route -> [reasoning -> summarize|general]
        workflow.add_edge("rag_retriever", "router")
        
        # Conditional routing based on document relevance scores
        workflow.add_conditional_edges(
            "router",
            self.router_agent.route_decision,
            {
                "relevant": "reasoning",  # Now goes to reasoning first
                "not_relevant": "general_response"
            }
        )
        
        # New flow: reasoning -> summarizer
        workflow.add_edge("reasoning", "summarizer")
        
        # Conditional edge after summarizer, based on a parameter/env variable
        if self.final_eval == "succinct":
            workflow.add_edge("summarizer", "final_answer")
            workflow.add_edge("final_answer", END)
            logger.info("Succinct evaluation mode enabled. Final answer node added.")
        else:
            workflow.add_edge("summarizer", END)
            logger.info("Verbose mode enabled. Final answer node bypassed.")
        
        # End points
        workflow.add_edge("general_response", END)
        
        logger.info("Workflow graph constructed: intent -> [retrieve -> route -> [reasoning -> summarize|general]] | general")
        return workflow
    
    def _generate_workflow_diagram(self):
        """Generate and save workflow diagram with tool information"""
        try:
            graph = self.app.get_graph()
            
            # Get tool information
            agent_tools = self._get_agent_tools()
            
            # REMOVED: Don't try to modify immutable graph nodes
            if agent_tools:
                logger.info(f"Including tools in workflow diagram: {agent_tools}")
            
            # Generate the diagram
            png_bytes = graph.draw_mermaid_png()
            out_path = "current_workflow.png"
            with open(out_path, "wb") as f:
                f.write(png_bytes)
            
            # Also save a text version with tool information
            self._save_workflow_text_with_tools(agent_tools)
            
            logger.info(f"Workflow diagram saved to {out_path}")
            if agent_tools:
                logger.info(f"Tool information saved to current_workflow_tools.txt")
                
        except Exception as e:
            logger.warning(f"Failed to generate workflow diagram: {e}")

    def _save_workflow_text_with_tools(self, agent_tools):
        """Save a text representation of the workflow with tool information"""
        try:
            with open("current_workflow_tools.txt", "w") as f:
                f.write("RAG WORKFLOW WITH TOOLS\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("WORKFLOW FLOW:\n")
                f.write("intent_classifier -> [rag_retriever -> router -> reasoning -> summarizer") 
                if self.final_eval == "succinct":
                    f.write(" -> final_answer")
                f.write("] | general_response\n\n")
                
                f.write("AGENT TOOLS:\n")
                f.write("-" * 30 + "\n")
                
                for agent_name, tools in agent_tools.items():
                    f.write(f"\n{agent_name.upper()}:\n")
                    for tool in tools:
                        f.write(f"  • {tool}\n")
                
                if not agent_tools:
                    f.write("No tools detected in agents\n")
                    
        except Exception as e:
            logger.warning(f"Failed to save workflow text with tools: {e}")

    def invoke(self, initial_state: ChatState) -> ChatState:
        """Execute the workflow with the given initial state"""
        print("\n" + "="*80)
        print("🚀 STARTING AGENTIC RAG WORKFLOW")
        print("="*80)
        print(f"📝 User Query: {initial_state['query']}")
        print("-" * 80)
        
        logger.info("Starting workflow execution")
        result = self.app.invoke(initial_state)
        
        print("\n" + "="*80)
        print("✅ WORKFLOW EXECUTION COMPLETED")
        print("="*80)
        
        # Show final results summary
        if result.get("is_corpus_relevant"):
            print(f"🎯 Intent Classification: CORPUS-RELEVANT")
        else:
            print(f"❌ Intent Classification: NOT CORPUS-RELEVANT")
            
        if result.get("is_relevant"):
            print(f"📊 Document Relevance: ABOVE THRESHOLD ({result.get('max_relevance_score', 0):.3f})")
        else:
            print(f"📊 Document Relevance: BELOW THRESHOLD ({result.get('max_relevance_score', 0):.3f})")
            
        print("\n🏁 FINAL OUTPUTS:")
        if result.get("reasoning_answer"):
            print(f"   🧠 Reasoning Answer: {len(result.get('reasoning_answer', ''))} characters")
        if result.get("summarized_answer"):
            print(f"   📝 Summarized Answer: {len(result.get('summarized_answer', ''))} characters")
        if result.get("final_answer"):
            print(f"   🎯 Final Answer: '{result.get('final_answer', '')}'")
        print("="*80)
        
        logger.info("Workflow execution completed")
        return result
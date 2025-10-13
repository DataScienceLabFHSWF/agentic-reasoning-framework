"""
reasoning_agent.py
Agent for deep reasoning over retrieved documents with ReAct loop capability
"""

import logging
from typing import Dict, Any, List, Tuple
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from .chat_state import ChatState
from .prompts import REASONING_PROMPT, REACT_REASONING_PROMPT
from .retriever_tool import RetrieverTool
import time
logger = logging.getLogger(__name__)


class ReasoningAgent:
    """Agent for performing deep reasoning over retrieved documents with ReAct loop"""
    
    def __init__(
        self, 
        llm: BaseChatModel, 
        chroma_dir: str,
        processed_dir: str,
        max_iterations: int = 3,
        relevance_threshold: float = 0.3
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold
        
        # Initialize retriever tool
        self.retriever_tool = RetrieverTool(
            chroma_dir=chroma_dir,
            processed_dir=processed_dir,
            relevance_threshold=relevance_threshold
        )
        
        # Setup tools using test_tool_binding approach
        self.llm_with_tools = self._setup_tools()
        
        logger.info(f"Reasoning agent initialized with max_iterations={max_iterations}")
    
    def _setup_tools(self):
        """Setup tools using the test_tool_binding approach"""
        try:
            # Get the tool
            tools = [self.retriever_tool.as_langchain_tool()]
            print(f"🔧 Created tools: {[tool.name for tool in tools]}")
            
            # Bind tools directly to LLM (like in test_tool_binding)
            llm_with_tools = self.llm.bind_tools(tools)
            
            logger.info(f"Successfully bound tools to LLM: {type(self.llm).__name__}")
            print("✅ Tools bound successfully to LLM")
            
            return llm_with_tools
                
        except Exception as e:
            logger.error(f"Failed to bind tools to LLM: {e}")
            print(f"❌ Failed to bind tools: {e}")
            return None

    def reason_over_documents(self, state: ChatState) -> Dict[str, Any]:
        """Perform deep reasoning analysis with optional ReAct loop"""
        query = state["query"]
        initial_docs = state.get("retrieved_docs", [])
        
        print("\n🧠 REASONING AGENT (ReAct)")
        print("-" * 50)
        print(f"📖 Initial documents: {len(initial_docs)}")
        print(f"🔄 Max iterations: {self.max_iterations}")
        
        logger.info(f"Starting ReAct reasoning for query: '{query[:50]}...'")
        
        if not initial_docs:
            reasoning_answer = "Keine Dokumente verfügbar für die Analyse."
            print("❌ No documents to analyze")
            return self._create_response(state, reasoning_answer, [], [])
        
        if not self.llm_with_tools:
            reasoning_answer = "Tool binding nicht verfügbar."
            print("❌ Tool binding not available")
            return self._create_response(state, reasoning_answer, initial_docs, [])
        
        # Initialize ReAct loop variables
        current_context = self._format_documents(initial_docs)
        all_retrieved_docs = initial_docs.copy()
        followup_questions = []
        iteration = 0
        
        print(f"📄 Initial context length: {len(current_context)} characters")
        
        # Initialize enhanced tracking if not present
        if "tool_calls" not in state:
            state["tool_calls"] = []
        if "follow_up_questions" not in state:
            state["follow_up_questions"] = []
        if "additional_context" not in state:
            state["additional_context"] = []
        if "workflow_metadata" not in state:
            state["workflow_metadata"] = {}
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                print(f"\n🔄 Iteration {iteration}/{self.max_iterations}")
                
                # Use tool-calling approach only
                print("🔧 Using tool-calling approach with bound LLM")
                result = self._react_with_tools(query, current_context, iteration, followup_questions)
                
                # Parse result
                if result["needs_more_info"]:
                    followup_question = result["followup_question"]
                    print(f"🤔 Generated follow-up: {followup_question}")
                    followup_questions.append(followup_question)
                    
                    # FIXED: Track the tool call properly
                    tool_start_time = time.time()
                    
                    # Retrieve additional documents
                    new_docs = self.retriever_tool.retrieve(followup_question)
                    
                    tool_execution_time = time.time() - tool_start_time
                    
                    # Track this tool call
                    state["tool_calls"].append({
                        "name": "retrieve_documents",
                        "tool_name": "retrieve_documents",
                        "args": {"query": followup_question},
                        "input": {"query": followup_question},
                        "output": f"Retrieved {len(new_docs)} documents",
                        "success": len(new_docs) > 0,
                        "execution_time": tool_execution_time,
                        "timestamp": time.time(),
                        "iteration": iteration
                    })
                    
                    if new_docs:
                        print(f"📚 Retrieved {len(new_docs)} additional documents")
                        # Add new documents to context
                        new_context = self._format_documents(new_docs, is_additional=True)
                        current_context += f"\n\nZUSÄTZLICHE INFORMATIONEN (Iteration {iteration}):\n{new_context}"
                        all_retrieved_docs.extend(new_docs)
                    else:
                        print("❌ No additional documents found")
                        # No new information, exit loop
                        break
                else:
                    # Got a satisfactory answer
                    print(f"✅ Satisfactory answer achieved in {iteration} iterations")
                    reasoning_answer = result["answer"]
                    break
            else:
                # Max iterations reached
                print(f"⏰ Max iterations ({self.max_iterations}) reached")
                # Generate final answer with available context
                reasoning_answer = self._generate_final_answer(query, current_context)
            
            print(f"📝 Final answer length: {len(reasoning_answer)} characters")
            print(f"🔍 Total documents used: {len(all_retrieved_docs)}")
            print("→→→ Proceeding to SUMMARIZER AGENT...")
            
            logger.info(f"ReAct reasoning completed: {iteration} iterations, {len(all_retrieved_docs)} total docs")
            
        except Exception as e:
            logger.error(f"ReAct reasoning error: {e}")
            reasoning_answer = f"Fehler beim Reasoning: {str(e)}"
            print(f"❌ Reasoning Error: {str(e)}")
        
        return self._create_response(state, reasoning_answer, all_retrieved_docs, followup_questions)
    
    def _react_with_tools(self, query: str, context: str, iteration: int, followup_questions: List[str] = None) -> Dict[str, Any]:
        """ReAct reasoning using tool-calling approach with bound LLM"""
        
        # Use existing prompt from prompts.py
        messages = REACT_REASONING_PROMPT.format_messages(
            query=query,
            context=context,
            iteration=iteration,
            previous_followups=followup_questions or []
        )
        
        print(f"🔍 Query: {query}")
        print(f"📝 Context Length: {len(context)} characters")
        print(f"🔄 Iteration: {iteration}")
        if followup_questions:
            print(f"🤔 Previous follow-ups: {followup_questions}")
        
        try:
            # Invoke the LLM with bound tools
            print("🚀 Invoking LLM with bound tools...")
            response = self.llm_with_tools.invoke(messages)
            
            print(f"📤 LLM Response Content: {response.content[:200]}...")
            print(f"🔧 Tool Calls Present: {bool(response.tool_calls)}")
            
            # Check if tools were called
            if response.tool_calls:
                # Model wants more information
                tool_call = response.tool_calls[0]
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                # Extract follow-up question from different possible arg structures
                followup_question = ""
                if "query" in tool_args:
                    followup_question = tool_args["query"]
                elif "__arg1" in tool_args:
                    followup_question = tool_args["__arg1"]
                elif len(tool_args) == 1:
                    # Take the first value if there's only one arg
                    followup_question = list(tool_args.values())[0]
                
                print(f"🔧 Tool Call Detected!")
                print(f"   Tool Name: {tool_name}")
                print(f"   Tool Args: {tool_args}")
                print(f"🤔 Follow-up Question: {followup_question}")
                
                return {
                    "needs_more_info": True,
                    "followup_question": followup_question,
                    "answer": ""
                }
            else:
                # Model provided direct answer
                print("✅ No Tool Calls - Model provided direct answer")
                print(f"📝 Answer Preview: {response.content[:200]}...")
                
                return {
                    "needs_more_info": False,
                    "followup_question": "",
                    "answer": response.content
                }
                
        except Exception as e:
            logger.error(f"Tool-based ReAct error: {e}")
            print(f"❌ Error during tool-based reasoning: {e}")
            # Return error as final answer
            return {
                "needs_more_info": False,
                "followup_question": "",
                "answer": f"Fehler beim Reasoning: {str(e)}"
            }
    
    def _generate_final_answer(self, query: str, context: str) -> str:
        """Generate final answer when max iterations reached"""
        messages = REASONING_PROMPT.format_messages(query=query, context=context)
        response = self.llm.invoke(messages)
        return response.content
    
    def _format_documents(self, docs: List[Any], is_additional: bool = False) -> str:
        """Format documents for context"""
        context = ""
        
        if is_additional:
            context += "=== DURCH TOOL-AUFRUF ABGERUFENE ZUSÄTZLICHE DOKUMENTE ===\n"
            context += "HINWEIS: Diese Dokumente wurden durch vorherige Tool-Aufrufe abgerufen.\n\n"
        
        for i, doc in enumerate(docs, 1):
            source = getattr(doc, 'metadata', {}).get('filename', f'Dokument {i}')
            content = doc.page_content
            score = getattr(doc, 'metadata', {}).get('score', 0.0)
            
            prefix = "ZUSÄTZLICH " if is_additional else ""
            context += f"{prefix}Dokument {i} ({source}, Score: {score:.3f}):\n{content}\n\n"
        
        return context
    
    def _create_response(
        self, 
        state: ChatState, 
        reasoning_answer: str, 
        all_docs: List[Any], 
        followup_questions: List[str]
    ) -> Dict[str, Any]:
        """Create the response state with proper tracking"""
        
        # Calculate additional context count
        initial_docs_count = len(state.get("retrieved_docs", []))
        additional_retrieved_context = len(all_docs) - initial_docs_count
        
        # Initialize tracking fields if not present
        if "tool_calls" not in state:
            state["tool_calls"] = []
        if "follow_up_questions" not in state:
            state["follow_up_questions"] = []
        if "additional_context" not in state:
            state["additional_context"] = []
        if "workflow_metadata" not in state:
            state["workflow_metadata"] = {}
        
        # Add follow-up questions to state tracking
        for question in followup_questions:
            state["follow_up_questions"].append(question)
        
        # Add additional context information for any new documents
        if additional_retrieved_context > 0:
            new_docs = all_docs[initial_docs_count:]  # Get only the new documents
            for doc in new_docs:
                state["additional_context"].append({
                    "source": doc.metadata.get('filename', 'unknown'),
                    "content": doc.page_content,
                    "relevance_score": doc.metadata.get('score', 0.0),
                    "context_type": "follow_up_retrieval",
                    "retrieved_by": "reasoning_agent"
                })
        
        # Update workflow metadata
        state["workflow_metadata"].update({
            "total_follow_up_questions": len(state["follow_up_questions"]),
            "total_additional_context": len(state["additional_context"]),
            "additional_retrieved_context": additional_retrieved_context
        })
        
        return {
            **state,
            "reasoning_answer": reasoning_answer,
            "retrieved_docs": all_docs,  # Update with all retrieved docs
            "followup_questions": followup_questions,  # Keep for backward compatibility
            "additional_retrieved_context": additional_retrieved_context  # Keep for backward compatibility
        }
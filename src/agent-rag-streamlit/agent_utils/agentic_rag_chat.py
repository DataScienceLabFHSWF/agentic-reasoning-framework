"""
agentic_rag_chat.py
Main RAG chat class that orchestrates the workflow
"""

import logging
import os
from typing import List, Dict
from langchain_core.messages import HumanMessage

from .chat_state import ChatState
from .workflow import RAGWorkflow
from .router_agent import RouterAgent
from .retriever_agent import RetrieverAgent
from .summarizer_agent import SummarizerAgent
from .general_agent import GeneralAgent
from .final_answer_agent import FinalAnswerAgent # New import
from .model_factory import create_model
from .intent_agent import IntentClassificationAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgenticRAGChat:
    """
    Main RAG chat class that orchestrates the entire workflow with intent classification
    """
    
    def __init__(
        self,
        chroma_dir: str,
        processed_dir: str,
        intent_model: str = "llama3.1:latest",
        router_model: str = "llama3.1:latest",
        summarizer_model: str = "llama3.1:latest",
        general_model: str = "llama3.1:latest",
        final_answer_model: str = "llama3.1:latest", # New parameter
        temperature: float = 0.0,
        retrieval_k: int = 5,
        relevance_threshold: float = 0.05,
        force_german: bool = True  # New parameter
    ):
        self.chroma_dir = chroma_dir
        self.processed_dir = processed_dir
        self.chat_history = []
        self.relevance_threshold = relevance_threshold
        self.force_german = force_german

        logger.info("Initializing Agentic RAG Chat")
        logger.info(f"Intent model: {intent_model}")
        logger.info(f"Router model: {router_model}")
        logger.info(f"Summarizer model: {summarizer_model}")
        logger.info(f"General model: {general_model}")
        logger.info(f"Final answer model: {final_answer_model}")
        logger.info(f"Relevance threshold: {relevance_threshold}")
        
        # Preload all models into memory
        logger.info("Loading models into memory...")
        self.intent_llm = self._preload_model(intent_model, temperature, "Intent")
        self.router_llm = self._preload_model(router_model, temperature, "Router")
        self.summarizer_llm = self._preload_model(summarizer_model, temperature, "Summarizer")
        self.general_llm = self._preload_model(general_model, temperature, "General")
        self.final_answer_llm = self._preload_model(final_answer_model, temperature, "Final Answer") # New model loading
        
        # Initialize agents
        logger.info("Initializing agents...")
        self.intent_agent = IntentClassificationAgent(self.intent_llm)
        self.router_agent = RouterAgent(self.router_llm, relevance_threshold)
        self.retriever_agent = RetrieverAgent(chroma_dir, processed_dir, k=retrieval_k)
        self.summarizer_agent = SummarizerAgent(self.summarizer_llm)
        self.general_agent = GeneralAgent(self.general_llm)
        self.final_answer_agent = FinalAnswerAgent(self.final_answer_llm) # New agent initialization
        
        # Initialize workflow
        logger.info("Building workflow...")
        self.workflow = RAGWorkflow(
            intent_agent=self.intent_agent,
            router_agent=self.router_agent,
            retriever_agent=self.retriever_agent,
            summarizer_agent=self.summarizer_agent,
            general_agent=self.general_agent,
            final_answer_agent=self.final_answer_agent # Pass the new agent to workflow
        )
        
        logger.info("All models loaded and ready. Agentic RAG Chat initialized successfully")
    
    def _preload_model(self, model_name: str, temperature: float, model_type: str):
        """Preload model and test with a simple query to ensure it's ready"""
        logger.info(f"Loading {model_type} model: {model_name}")
        model = create_model(model=model_name, temperature=temperature)
        
        # Test the model with a simple query to ensure it's loaded
        try:
            logger.info(f"Testing {model_type} model...")
            test_response = model.invoke([{"role": "user", "content": "Hello"}])
            logger.info(f"{model_type} model loaded and ready")
            return model
        except Exception as e:
            logger.error(f"Failed to load {model_type} model: {e}")
            raise e
    
    def chat(self, user_input: str) -> str:
        """
        Process a single chat message and return the response
        
        Args:
            user_input: The user's message/query
            
        Returns:
            The AI's response
        """
        logger.info(f"Processing user input: '{user_input[:50]}...'")
        
        # Create initial state
        initial_state = ChatState(
            messages=[HumanMessage(content=user_input)],
            query=user_input,
            is_corpus_relevant=None,  # New field
            intent_reasoning=None,    # New field
            is_relevant=False,
            retrieved_docs=[],
            max_relevance_score=0.0,
            summarized_answer=None,
            final_answer=None,
            chat_history=self.chat_history
        )
        
        try:
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract the answer - prefer final_answer if available, otherwise summarized_answer
            answer = result.get("final_answer") or result.get("summarized_answer", "")
            
            # Update chat history
            self.chat_history.append({"user": user_input, "assistant": answer})
            
            # Keep only last 10 exchanges to manage memory
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
                logger.info("Chat history trimmed to last 10 exchanges")
            
            logger.info("Chat response generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def start_chat(self):
        """Start an interactive chat loop"""
        logger.info("Starting interactive chat session")
        print("RAG Chat Assistant is ready! Type 'quit', 'exit', or 'bye' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                # Check for exit conditions
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("Goodbye! Thanks for chatting!")
                    logger.info("Chat session ended by user")
                    break
                
                if not user_input:
                    print("Please enter a message.")
                    continue
                
                # Get response
                print("Assistant: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                print()  # Add a blank line for readability
                
            except KeyboardInterrupt:
                print("\nChat interrupted. Goodbye!")
                logger.info("Chat session interrupted")
                break
            except Exception as e:
                logger.error(f"Chat session error: {e}")
                print(f"Error: {e}")
                print("Please try again.")
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history"""
        return self.chat_history.copy()
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history.clear()
        logger.info("Chat history cleared")

def create_rag_chat(
    chroma_dir: str,
    processed_dir: str,
    intent_model: str = "llama3.1:latest",
    router_model: str = "gpt-oss:20b",
    summarizer_model: str = "llama3.1:latest",
    general_model: str = "llama3.1:latest",
    final_answer_model: str = "llama3.1:latest", # New parameter
    temperature: float = 0.0,
    retrieval_k: int = 3,
    relevance_threshold: float = 0.5
) -> AgenticRAGChat:
    """
    Create and return a RAG chat instance with intent classification
    
    Args:
        chroma_dir: Path to ChromaDB directory
        processed_dir: Path to processed documents directory
        intent_model: Model for intent classification
        router_model: Model for routing decisions
        summarizer_model: Model for summarization
        general_model: Model for general responses
        final_answer_model: Model for the succinct final answer
        temperature: Temperature for response generation
        retrieval_k: Number of documents to retrieve
        relevance_threshold: Minimum relevance score to use RAG
        
    Returns:
        Configured AgenticRAGChat instance
    """
    return AgenticRAGChat(
        chroma_dir=chroma_dir,
        processed_dir=processed_dir,
        intent_model=intent_model,
        router_model=router_model,
        summarizer_model=summarizer_model,
        general_model=general_model,
        final_answer_model=final_answer_model,
        temperature=temperature,
        retrieval_k=retrieval_k,
        relevance_threshold=relevance_threshold
    )


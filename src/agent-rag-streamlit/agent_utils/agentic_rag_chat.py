"""
agentic_rag_chat.py
Main RAG chat class that orchestrates the workflow
"""

import logging
from typing import List, Dict
from langchain_core.messages import HumanMessage

from .chat_state import ChatState
from .workflow import RAGWorkflow
from .router_agent import RouterAgent
from .retriever_agent import RetrieverAgent
from .summarizer_agent import SummarizerAgent
from .general_agent import GeneralAgent
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
        temperature: float = 0.0,
        retrieval_k: int = 3,
        relevance_threshold: float = 0.5,
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
        logger.info(f"Relevance threshold: {relevance_threshold}")
        
        # Preload all models into memory
        logger.info("Loading models into memory...")
        self.intent_llm = self._preload_model(intent_model, temperature, "Intent")
        self.router_llm = self._preload_model(router_model, temperature, "Router")
        self.summarizer_llm = self._preload_model(summarizer_model, temperature, "Summarizer")
        self.general_llm = self._preload_model(general_model, temperature, "General")
        
        # Initialize agents
        logger.info("Initializing agents...")
        self.intent_agent = IntentClassificationAgent(self.intent_llm)
        self.router_agent = RouterAgent(self.router_llm, relevance_threshold)
        self.retriever_agent = RetrieverAgent(chroma_dir, processed_dir, k=retrieval_k)
        self.summarizer_agent = SummarizerAgent(self.summarizer_llm)
        self.general_agent = GeneralAgent(self.general_llm)
        
        # Initialize workflow
        logger.info("Building workflow...")
        self.workflow = RAGWorkflow(
            intent_agent=self.intent_agent,
            router_agent=self.router_agent,
            retriever_agent=self.retriever_agent,
            summarizer_agent=self.summarizer_agent,
            general_agent=self.general_agent
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
            answer="",
            chat_history=self.chat_history
        )
        
        try:
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            # Extract the answer
            answer = result["answer"]
            
            # Enforce German response if enabled
            if self.force_german and not self._is_german_response(answer):
                logger.info("Enforcing German response")
                answer = self._ensure_german_response(answer, user_input)
            
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
    
    def _is_german_response(self, text: str) -> bool:
        """Check if response is primarily in German"""
        german_indicators = ['der', 'die', 'das', 'und', 'oder', 'aber', 'ich', 'Sie', 'wir', 'sind', 'haben', 'können', 'Kernkraftwerk', 'Genehmigung', 'Anlage']
        words = text.lower().split()
        if len(words) < 5:
            return True
        german_count = sum(1 for word in words if any(indicator in word for indicator in german_indicators))
        return german_count / len(words) > 0.25

    def _ensure_german_response(self, response: str, query: str) -> str:
        """Ensure response is in German"""
        try:
            german_prompt = f"""
            Bitte antworte auf die folgende Frage ausschließlich auf Deutsch:
            
            Frage: {query}
            
            Falls bereits eine Antwort vorhanden ist, übersetze sie ins Deutsche:
            {response}
            
            Deutsche Antwort:
            """
            german_response = self.general_llm.invoke([{"role": "user", "content": german_prompt}])
            return german_response.content
        except Exception as e:
            logger.error(f"German enforcement failed: {e}")
            return f"Entschuldigung, es gab einen Fehler bei der Verarbeitung Ihrer Anfrage."


def create_rag_chat(
    chroma_dir: str,
    processed_dir: str,
    intent_model: str = "llama3.1:latest",
    router_model: str = "gpt-oss:20b",
    summarizer_model: str = "llama3.1:latest",
    general_model: str = "llama3.1:latest",
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
        temperature=temperature,
        retrieval_k=retrieval_k,
        relevance_threshold=relevance_threshold
    )
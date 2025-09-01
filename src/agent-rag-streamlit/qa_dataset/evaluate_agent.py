"""
evaluate_agent.py
Evaluation script for testing the agent workflow on QA datasets
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

# Add the parent directory to the path to import agent_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_utils import create_rag_chat, AgenticRAGChat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Evaluator for testing agent workflow on QA datasets"""
    
    def __init__(
        self,
        chroma_dir: str,
        processed_dir: str,
        qa_dataset_path: str,
        results_output_path: str = None,
        **agent_kwargs
    ):
        # Validate paths
        if not chroma_dir or not os.path.exists(chroma_dir):
            raise ValueError(f"ChromaDB directory does not exist or is None: {chroma_dir}")
        
        if not processed_dir or not os.path.exists(processed_dir):
            raise ValueError(f"Processed documents directory does not exist or is None: {processed_dir}")
        
        if not qa_dataset_path or not os.path.exists(qa_dataset_path):
            raise ValueError(f"QA dataset file does not exist or is None: {qa_dataset_path}")
        
        self.chroma_dir = chroma_dir
        self.processed_dir = processed_dir
        self.qa_dataset_path = qa_dataset_path
        
        logger.info(f"ChromaDB directory: {chroma_dir}")
        logger.info(f"Processed documents directory: {processed_dir}")
        logger.info(f"QA dataset path: {qa_dataset_path}")
        
        # Default output path if not provided - FIXED
        if results_output_path is None or results_output_path.strip() == "":
            base_name = os.path.splitext(os.path.basename(qa_dataset_path))[0]
            results_dir = os.path.dirname(qa_dataset_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_output_path = os.path.join(results_dir, f"{base_name}_agent_results_{timestamp}.json")
        else:
            self.results_output_path = results_output_path
            
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.results_output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        
        logger.info(f"Results will be saved to: {self.results_output_path}")
        
        # Initialize the RAG chat agent
        logger.info("Initializing RAG chat agent for evaluation...")
        try:
            self.agent = create_rag_chat(
                chroma_dir=chroma_dir,
                processed_dir=processed_dir,
                **agent_kwargs
            )
            logger.info("Agent initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise e
    
    def load_qa_dataset(self) -> Dict[str, Any]:
        """Load the QA dataset from JSON file"""
        logger.info(f"Loading QA dataset from: {self.qa_dataset_path}")
        
        with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} datasets")
        return dataset
    
    def evaluate_single_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question through the agent workflow"""
        question = question_data["question"]
        expected_answer = question_data["answer"]
        
        logger.info(f"Evaluating: {question[:50]}...")
        
        start_time = time.time()
        
        # Clear chat history before each question to ensure clean state
        self.agent.clear_chat_history()
        
        # Create initial state and run workflow
        try:
            # We need to access the workflow directly to get all answers
            from langchain_core.messages import HumanMessage
            from agent_utils.chat_state import ChatState
            
            initial_state = ChatState(
                messages=[HumanMessage(content=question)],
                query=question,
                is_corpus_relevant=None,
                intent_reasoning=None,
                is_relevant=False,
                retrieved_docs=[],
                max_relevance_score=0.0,
                reasoning_answer=None,  # Added reasoning answer field
                summarized_answer=None,
                final_answer=None,
                chat_history=[]
            )
            
            # Run the workflow
            result = self.agent.workflow.invoke(initial_state)
            
            # Extract all answers and metadata from the workflow
            reasoning_answer = result.get("reasoning_answer", "")
            summarized_answer = result.get("summarized_answer", "")
            final_answer = result.get("final_answer", "")
            is_corpus_relevant = result.get("is_corpus_relevant", None)
            intent_reasoning = result.get("intent_reasoning", "")
            
            # Additional metadata from workflow
            is_relevant = result.get("is_relevant", None)
            max_relevance_score = result.get("max_relevance_score", 0.0)
            num_retrieved_docs = len(result.get("retrieved_docs", []))
            
            processing_time = time.time() - start_time
            
            logger.info(f"Question processed in {processing_time:.2f}s")
            logger.info(f"Intent classification: {'corpus-relevant' if is_corpus_relevant else 'general'}")
            logger.info(f"Reasoning answer: {len(reasoning_answer)} characters")
            logger.info(f"Summarized answer: {summarized_answer[:100]}...")
            logger.info(f"Final answer: {final_answer}")
            
            return {
                **question_data,  # Include original question data
                "agent_responses": {
                    "reasoning_answer": reasoning_answer,
                    "summarized_answer": summarized_answer,
                    "final_answer": final_answer
                },
                "intent_classification": {
                    "is_corpus_relevant": is_corpus_relevant,
                    "intent_reasoning": intent_reasoning
                },
                "workflow_metadata": {
                    "is_relevant": is_relevant,
                    "max_relevance_score": max_relevance_score,
                    "num_retrieved_docs": num_retrieved_docs,
                    "processing_time_seconds": processing_time
                },
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing question: {e}")
            
            return {
                **question_data,
                "agent_responses": {
                    "reasoning_answer": f"ERROR: {str(e)}",
                    "summarized_answer": f"ERROR: {str(e)}",
                    "final_answer": f"ERROR: {str(e)}"
                },
                "intent_classification": {
                    "is_corpus_relevant": None,
                    "intent_reasoning": f"Error during processing: {str(e)}"
                },
                "workflow_metadata": {
                    "is_relevant": None,
                    "max_relevance_score": 0.0,
                    "num_retrieved_docs": 0,
                    "processing_time_seconds": processing_time,
                    "error": str(e)
                },
                "evaluation_timestamp": datetime.now().isoformat()
            }
    
    def evaluate_dataset(self, dataset_name: str, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all questions in a dataset"""
        logger.info(f"Evaluating dataset: {dataset_name} ({len(questions)} questions)")
        
        results = []
        total_start_time = time.time()
        
        for i, question_data in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)} in {dataset_name}")
            result = self.evaluate_single_question(question_data)
            results.append(result)
            
            # Brief pause between questions to avoid overwhelming the system
            time.sleep(0.5)
        
        total_processing_time = time.time() - total_start_time
        
        # Calculate summary statistics
        successful_questions = [r for r in results if not r["workflow_metadata"].get("error")]
        error_count = len(results) - len(successful_questions)
        avg_processing_time = sum(r["workflow_metadata"]["processing_time_seconds"] for r in results) / len(results)
        
        dataset_summary = {
            "dataset_name": dataset_name,
            "total_questions": len(questions),
            "successful_questions": len(successful_questions),
            "error_count": error_count,
            "total_processing_time_seconds": total_processing_time,
            "average_processing_time_seconds": avg_processing_time
        }
        
        logger.info(f"Dataset {dataset_name} evaluation complete:")
        logger.info(f"  - {len(successful_questions)}/{len(questions)} questions processed successfully")
        logger.info(f"  - Average processing time: {avg_processing_time:.2f}s")
        logger.info(f"  - Total time: {total_processing_time:.2f}s")
        
        return {
            "summary": dataset_summary,
            "results": results
        }
    
    def run_evaluation(self, datasets_to_evaluate: List[str] = None) -> Dict[str, Any]:
        """Run evaluation on specified datasets (default: dataset1, dataset2, dataset3)"""
        if datasets_to_evaluate is None:
            datasets_to_evaluate = ["dataset1", "dataset2", "dataset3"]
        
        qa_data = self.load_qa_dataset()
        
        # Evaluation metadata
        evaluation_metadata = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "qa_dataset_path": self.qa_dataset_path,
            "chroma_dir": self.chroma_dir,
            "processed_dir": self.processed_dir,
            "datasets_evaluated": datasets_to_evaluate,
            "agent_config": {
                "relevance_threshold": self.agent.relevance_threshold,
                "retrieval_k": self.agent.retriever_agent.k
            }
        }
        
        # Run evaluation on each specified dataset
        evaluation_results = {}
        overall_start_time = time.time()
        
        for dataset_name in datasets_to_evaluate:
            if dataset_name in qa_data:
                logger.info(f"Starting evaluation of {dataset_name}")
                evaluation_results[dataset_name] = self.evaluate_dataset(dataset_name, qa_data[dataset_name])
            else:
                logger.warning(f"Dataset {dataset_name} not found in QA data")
                evaluation_results[dataset_name] = {
                    "summary": {"error": f"Dataset {dataset_name} not found"},
                    "results": []
                }
        
        overall_processing_time = time.time() - overall_start_time
        evaluation_metadata["total_evaluation_time_seconds"] = overall_processing_time
        
        # Combine all results
        final_results = {
            "metadata": evaluation_metadata,
            "evaluation_results": evaluation_results
        }
        
        # Save results
        self.save_results(final_results)
        
        return final_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON file with robust error handling"""
        try:
            # Double-check the output path is valid
            if not self.results_output_path or self.results_output_path.strip() == "":
                # Fallback to current directory if somehow the path is empty
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.results_output_path = f"agent_evaluation_results_{timestamp}.json"
                logger.warning(f"Output path was empty, using fallback: {self.results_output_path}")
            
            # Ensure directory exists
            output_dir = os.path.dirname(self.results_output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created directory: {output_dir}")
            
            # If no directory in path, save to current directory
            if not output_dir:
                current_dir = os.getcwd()
                filename = os.path.basename(self.results_output_path)
                self.results_output_path = os.path.join(current_dir, filename)
                logger.info(f"Saving to current directory: {self.results_output_path}")
            
            logger.info(f"Saving evaluation results to: {self.results_output_path}")
            
            # Save the file
            with open(self.results_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Verify the file was created
            if os.path.exists(self.results_output_path):
                file_size = os.path.getsize(self.results_output_path)
                logger.info(f"Results saved successfully! File size: {file_size} bytes")
                print(f"✅ Results saved successfully to: {self.results_output_path}")
                print(f"📁 File size: {file_size:,} bytes")
            else:
                raise FileNotFoundError(f"File was not created: {self.results_output_path}")
                
        except Exception as e:
            # Last resort: save to current directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_path = f"emergency_results_backup_{timestamp}.json"
            
            logger.error(f"Failed to save to {self.results_output_path}: {e}")
            logger.info(f"Attempting emergency backup to: {fallback_path}")
            
            try:
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                self.results_output_path = fallback_path
                logger.info(f"Emergency backup saved successfully to: {fallback_path}")
                print(f"⚠️  Original save failed, but emergency backup created: {fallback_path}")
                
            except Exception as backup_error:
                logger.error(f"Emergency backup also failed: {backup_error}")
                print(f"❌ Critical error: Could not save results anywhere!")
                print(f"Original error: {e}")
                print(f"Backup error: {backup_error}")
                raise backup_error
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a summary of evaluation results"""
        print("\n" + "="*60)
        print("AGENT EVALUATION SUMMARY")
        print("="*60)
        
        metadata = results["metadata"]
        print(f"Evaluation completed: {metadata['evaluation_timestamp']}")
        print(f"Total evaluation time: {metadata['total_evaluation_time_seconds']:.2f}s")
        print(f"Datasets evaluated: {', '.join(metadata['datasets_evaluated'])}")
        
        for dataset_name, dataset_results in results["evaluation_results"].items():
            if "error" in dataset_results["summary"]:
                print(f"\n{dataset_name}: ERROR - {dataset_results['summary']['error']}")
                continue
                
            summary = dataset_results["summary"]
            print(f"\n{dataset_name}:")
            print(f"  Questions: {summary['successful_questions']}/{summary['total_questions']} successful")
            print(f"  Avg time: {summary['average_processing_time_seconds']:.2f}s per question")
            print(f"  Total time: {summary['total_processing_time_seconds']:.2f}s")


def main():
    """Main function for running the evaluation"""
    # Configuration - adjust these paths according to your setup
    CHROMA_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/chroma_db"  # Update this path
    PROCESSED_DIR = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files"  # Update this path
    QA_DATASET_PATH = "/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/qa_dataset/gpt_qa_datasets_de.json"
    
    # Agent configuration - Updated to include reasoning agent
    agent_config = {
        "intent_model": "llama3.1:latest",
        "router_model": "llama3.1:latest", 
        "reasoning_model": "qwen3:14b",  # Added reasoning model
        "summarizer_model": "llama3.1:latest",
        "general_model": "llama3.1:latest",
        "final_answer_model": "llama3.1:latest",
        "temperature": 0.0,
        "retrieval_k": 3,
        "relevance_threshold": 0.15
    }
    
    # Initialize evaluator
    evaluator = AgentEvaluator(
        chroma_dir=CHROMA_DIR,
        processed_dir=PROCESSED_DIR,
        qa_dataset_path=QA_DATASET_PATH,
        **agent_config
    )
    
    # Run evaluation on datasets 1, 2, and 3
    logger.info("Starting agent evaluation on QA datasets...")
    results = evaluator.run_evaluation(["dataset1", "dataset2", "dataset3"])
    
    # Print summary
    evaluator.print_summary(results)
    
    print(f"\nDetailed results saved to: {evaluator.results_output_path}")


if __name__ == "__main__":
    main()

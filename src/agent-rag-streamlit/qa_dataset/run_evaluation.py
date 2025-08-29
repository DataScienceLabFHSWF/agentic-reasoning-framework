"""
run_evaluation.py
Simple script to run agent evaluation with customizable parameters
"""

import os
import argparse
from evaluate_agent import AgentEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG agent on QA datasets")
    
    parser.add_argument("--chroma_dir", required=True, help="Path to ChromaDB directory")
    parser.add_argument("--processed_dir", required=True, help="Path to processed documents directory")
    parser.add_argument("--qa_dataset", default="gpt_qa_datasets_de.json", help="QA dataset filename")
    parser.add_argument("--datasets", nargs="+", default=["dataset1", "dataset2", "dataset3"], 
                       help="Datasets to evaluate (default: dataset1 dataset2 dataset3)")
    parser.add_argument("--output", help="Output file path (default: auto-generated)")
    parser.add_argument("--relevance_threshold", type=float, default=0.2, help="Relevance threshold")
    parser.add_argument("--retrieval_k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=0.0, help="Model temperature")
    
    args = parser.parse_args()
    
    # Validate required paths
    if not os.path.exists(args.chroma_dir):
        print(f"Error: ChromaDB directory does not exist: {args.chroma_dir}")
        return 1
    
    if not os.path.exists(args.processed_dir):
        print(f"Error: Processed documents directory does not exist: {args.processed_dir}")
        return 1
    
    # Build full path to QA dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    qa_dataset_path = os.path.join(script_dir, args.qa_dataset)
    
    if not os.path.exists(qa_dataset_path):
        print(f"Error: QA dataset file does not exist: {qa_dataset_path}")
        return 1
    
    print(f"Using ChromaDB directory: {args.chroma_dir}")
    print(f"Using processed documents directory: {args.processed_dir}")
    print(f"Using QA dataset: {qa_dataset_path}")
    
    # Agent configuration
    agent_config = {
        "intent_model": "llama3.1:latest",
        "router_model": "llama3.1:latest",
        "summarizer_model": "llama3.1:latest", 
        "general_model": "llama3.1:latest",
        "final_answer_model": "llama3.1:latest",
        "temperature": args.temperature,
        "retrieval_k": args.retrieval_k,
        "relevance_threshold": args.relevance_threshold
    }
    
    # Initialize evaluator
    try:
        evaluator = AgentEvaluator(
            chroma_dir=args.chroma_dir,
            processed_dir=args.processed_dir,
            qa_dataset_path=qa_dataset_path,
            results_output_path=args.output,
            **agent_config
        )
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return 1
    
    # Run evaluation
    print(f"Starting evaluation on datasets: {', '.join(args.datasets)}")
    try:
        results = evaluator.run_evaluation(args.datasets)
        
        # Print summary
        evaluator.print_summary(results)
        
        print(f"\nEvaluation complete! Results saved to: {evaluator.results_output_path}")
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1


if __name__ == "__main__":
    main()

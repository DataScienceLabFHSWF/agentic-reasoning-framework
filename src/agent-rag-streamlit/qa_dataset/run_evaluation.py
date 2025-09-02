"""
run_evaluation.py
Simple script to run agent evaluation with customizable parameters
"""

import os
import argparse
from evaluate_agent import AgentEvaluator
from datetime import datetime
import json


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
    parser.add_argument("--analyze", action="store_true", help="Run analysis after evaluation")
    
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
    
    # Handle output path - ensure it's in the same directory as the script if not specified
    if args.output:
        output_path = args.output
    else:
        # Default to same directory as the QA dataset
        base_name = os.path.splitext(os.path.basename(qa_dataset_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(script_dir, f"{base_name}_agent_results_{timestamp}.json")
    
    print(f"Results will be saved to: {output_path}")
    
    # Agent configuration
    agent_config = {
        "intent_model": "llama3.1:latest",
        "router_model": "llama3.1:latest",
        "reasoning_model": "qwen3:14b",  # Added reasoning model
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
            results_output_path=output_path,  # Pass the explicit output path
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
        
        # Verify file exists and show size
        if os.path.exists(evaluator.results_output_path):
            file_size = os.path.getsize(evaluator.results_output_path)
            print(f"📁 File verified: {file_size:,} bytes")
        else:
            print("⚠️  Warning: Results file not found after save!")
        
        # Run analysis if requested
        if args.analyze:
            print("\nRunning analysis...")
            try:
                from analyze_results import ResultsAnalyzer
                
                output_file = evaluator.results_output_path.replace('.json', '_analysis.xlsx')
                analyzer = ResultsAnalyzer(evaluator.results_output_path, qa_dataset_path)
                analyzer.analyze(output_file)
                print(f"Analysis complete! Results saved to: {output_file}")
            except Exception as e:
                print(f"Error during analysis: {e}")
        
        return 0
    except Exception as e:
        print(f"Error during evaluation: {e}")
        
        # Try to save any partial results if they exist
        try:
            if hasattr(evaluator, 'partial_results') and evaluator.partial_results:
                emergency_file = f"emergency_partial_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(emergency_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluator.partial_results, f, ensure_ascii=False, indent=2)
                print(f"💾 Partial results saved to: {emergency_file}")
        except:
            pass
            
        return 1


if __name__ == "__main__":
    main()

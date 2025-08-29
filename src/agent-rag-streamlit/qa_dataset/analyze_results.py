"""
analyze_results.py
Script to analyze agent evaluation results and generate comprehensive metrics with plots
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Any, List
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


class ResultsAnalyzer:
    """Analyzer for agent evaluation results"""
    
    def __init__(self, results_file: str, qa_dataset_file: str):
        self.results_file = results_file
        self.qa_dataset_file = qa_dataset_file
        self.results_data = self.load_results()
        self.qa_data = self.load_qa_dataset()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_results(self) -> Dict[str, Any]:
        """Load evaluation results from JSON file"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_qa_dataset(self) -> Dict[str, Any]:
        """Load QA dataset from JSON file"""
        with open(self.qa_dataset_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        if answer is None:
            return ""
        
        # Convert to string and strip whitespace
        answer = str(answer).strip().lower()
        
        # Remove common punctuation and extra whitespace
        answer = re.sub(r'[.,;:!?"\']', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer
    
    def extract_numeric_value(self, text: str) -> float:
        """Extract numeric value from text"""
        if text is None:
            return None
        
        # Look for numbers (including decimals)
        numbers = re.findall(r'\d+(?:\.\d+)?', str(text))
        if numbers:
            return float(numbers[0])
        return None
    
    def compare_answers(self, expected: Any, actual: str, question_type: str = "general") -> bool:
        """Compare expected and actual answers"""
        if actual is None:
            return False
        
        expected_norm = self.normalize_answer(str(expected))
        actual_norm = self.normalize_answer(actual)
        
        # Handle different answer types
        if question_type == "numeric":
            expected_num = self.extract_numeric_value(expected)
            actual_num = self.extract_numeric_value(actual)
            if expected_num is not None and actual_num is not None:
                return abs(expected_num - actual_num) < 0.01
        
        # Check for exact match
        if expected_norm == actual_norm:
            return True
        
        # Check if expected answer is contained in actual answer
        if expected_norm in actual_norm:
            return True
        
        # Check if actual answer is contained in expected answer
        if actual_norm in expected_norm:
            return True
        
        # For yes/no questions
        if expected_norm in ["ja", "yes", "nein", "no"]:
            if expected_norm in ["ja", "yes"] and any(word in actual_norm for word in ["ja", "yes", "richtig", "korrekt"]):
                return True
            if expected_norm in ["nein", "no"] and any(word in actual_norm for word in ["nein", "no", "nicht", "falsch"]):
                return True
        
        return False
    
    def analyze_single_result(self, result: Dict[str, Any], expected_answer: Any) -> Dict[str, Any]:
        """Analyze a single question result"""
        # Get both agent answers
        summarized_answer = result.get("agent_responses", {}).get("summarized_answer", "")
        final_answer = result.get("agent_responses", {}).get("final_answer", "")
        
        # Check accuracy for both answers
        summarized_correct = self.compare_answers(expected_answer, summarized_answer)
        final_correct = self.compare_answers(expected_answer, final_answer)
        
        # Intent classification accuracy
        is_corpus_relevant = result.get("intent_classification", {}).get("is_corpus_relevant")
        intent_correct = is_corpus_relevant is True  # All questions should be corpus-relevant
        
        # Processing time
        processing_time = result.get("workflow_metadata", {}).get("processing_time_seconds", 0)
        
        # Retrieval metrics
        is_relevant = result.get("workflow_metadata", {}).get("is_relevant", False)
        max_relevance_score = result.get("workflow_metadata", {}).get("max_relevance_score", 0)
        num_retrieved_docs = result.get("workflow_metadata", {}).get("num_retrieved_docs", 0)
        
        return {
            "summarized_correct": summarized_correct,
            "final_correct": final_correct,
            "intent_correct": intent_correct,
            "processing_time": processing_time,
            "is_relevant": is_relevant,
            "max_relevance_score": max_relevance_score,
            "num_retrieved_docs": num_retrieved_docs,
            "expected_answer": str(expected_answer),
            "summarized_answer": summarized_answer,
            "final_answer": final_answer
        }
    
    def generate_metrics(self) -> List[Dict[str, Any]]:
        """Generate comprehensive metrics"""
        all_metrics = []
        
        for dataset_name, dataset_results in self.results_data["evaluation_results"].items():
            if "error" in dataset_results.get("summary", {}):
                continue
                
            results = dataset_results.get("results", [])
            qa_questions = self.qa_data.get(dataset_name, [])
            
            for i, result in enumerate(results):
                if i < len(qa_questions):
                    expected_answer = qa_questions[i]["answer"]
                    level = qa_questions[i]["level"]
                    question = qa_questions[i]["question"]
                    
                    analysis = self.analyze_single_result(result, expected_answer)
                    analysis.update({
                        "dataset": dataset_name,
                        "level": level,
                        "question": question,
                        "question_index": i
                    })
                    all_metrics.append(analysis)
        
        return all_metrics
    
    def create_summary_tables(self, metrics: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Create summary tables"""
        df = pd.DataFrame(metrics)
        
        tables = {}
        
        # Overall summary
        overall_summary = {
            "Total Questions": len(df),
            "Summarized Answer Accuracy": f"{df['summarized_correct'].mean():.2%}",
            "Final Answer Accuracy": f"{df['final_correct'].mean():.2%}",
            "Intent Classification Accuracy": f"{df['intent_correct'].mean():.2%}",
            "Avg Processing Time (s)": f"{df['processing_time'].mean():.2f}",
            "Avg Relevance Score": f"{df['max_relevance_score'].mean():.3f}",
            "Successful Retrievals": f"{df['is_relevant'].mean():.2%}"
        }
        tables["Overall Summary"] = pd.DataFrame([overall_summary])
        
        # By Dataset
        dataset_summary = df.groupby('dataset').agg({
            'summarized_correct': ['count', 'mean'],
            'final_correct': 'mean',
            'intent_correct': 'mean',
            'processing_time': 'mean',
            'max_relevance_score': 'mean',
            'is_relevant': 'mean'
        }).round(3)
        
        dataset_summary.columns = [
            'Total Questions', 'Summarized Accuracy', 'Final Accuracy', 
            'Intent Accuracy', 'Avg Processing Time', 'Avg Relevance Score', 
            'Retrieval Success Rate'
        ]
        tables["By Dataset"] = dataset_summary
        
        # By Level
        level_summary = df.groupby('level').agg({
            'summarized_correct': ['count', 'mean'],
            'final_correct': 'mean',
            'intent_correct': 'mean',
            'processing_time': 'mean',
            'max_relevance_score': 'mean',
            'is_relevant': 'mean'
        }).round(3)
        
        level_summary.columns = [
            'Total Questions', 'Summarized Accuracy', 'Final Accuracy', 
            'Intent Accuracy', 'Avg Processing Time', 'Avg Relevance Score', 
            'Retrieval Success Rate'
        ]
        tables["By Level"] = level_summary
        
        # By Dataset and Level
        dataset_level_summary = df.groupby(['dataset', 'level']).agg({
            'summarized_correct': ['count', 'mean'],
            'final_correct': 'mean',
            'intent_correct': 'mean',
            'processing_time': 'mean'
        }).round(3)
        
        dataset_level_summary.columns = [
            'Total Questions', 'Summarized Accuracy', 'Final Accuracy', 
            'Intent Accuracy', 'Avg Processing Time'
        ]
        tables["By Dataset and Level"] = dataset_level_summary
        
        return tables
    
    def create_plots(self, metrics: List[Dict[str, Any]], output_dir: str = None) -> List[str]:
        """Create comprehensive plots and return list of plot files"""
        df = pd.DataFrame(metrics)
        plot_files = []
        
        if output_dir is None:
            output_dir = Path(self.results_file).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Accuracy Comparison Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # Accuracy by Dataset
        dataset_acc = df.groupby('dataset')[['summarized_correct', 'final_correct']].mean()
        dataset_acc.plot(kind='bar', ax=axes[0,0], title='Accuracy by Dataset')
        axes[0,0].set_ylabel('Accuracy Rate')
        axes[0,0].set_xlabel('Dataset')
        axes[0,0].legend(['Summarized Answer', 'Final Answer'])
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Accuracy by Level
        level_acc = df.groupby('level')[['summarized_correct', 'final_correct']].mean()
        level_acc.plot(kind='bar', ax=axes[0,1], title='Accuracy by Difficulty Level')
        axes[0,1].set_ylabel('Accuracy Rate')
        axes[0,1].set_xlabel('Difficulty Level')
        axes[0,1].legend(['Summarized Answer', 'Final Answer'])
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Processing Time Distribution
        axes[1,0].hist(df['processing_time'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Processing Time Distribution')
        axes[1,0].set_xlabel('Processing Time (seconds)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].axvline(df['processing_time'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["processing_time"].mean():.2f}s')
        axes[1,0].legend()
        
        # Relevance Score Distribution
        axes[1,1].hist(df['max_relevance_score'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].set_title('Document Relevance Score Distribution')
        axes[1,1].set_xlabel('Max Relevance Score')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(df['max_relevance_score'].mean(), color='red', linestyle='--',
                         label=f'Mean: {df["max_relevance_score"].mean():.3f}')
        axes[1,1].legend()
        
        plt.tight_layout()
        plot_file = output_dir / 'performance_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        # 2. Heatmap of Accuracy by Dataset and Level
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pivot_data = df.pivot_table(values='final_correct', index='level', 
                                   columns='dataset', aggfunc='mean')
        
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0.5, 
                   ax=ax, fmt='.2f', cbar_kws={'label': 'Accuracy Rate'})
        ax.set_title('Final Answer Accuracy: Dataset vs Difficulty Level', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Difficulty Level')
        
        plt.tight_layout()
        plot_file = output_dir / 'accuracy_heatmap.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        # 3. Processing Time Analysis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Processing time by dataset
        df.boxplot(column='processing_time', by='dataset', ax=axes[0])
        axes[0].set_title('Processing Time by Dataset')
        axes[0].set_xlabel('Dataset')
        axes[0].set_ylabel('Processing Time (seconds)')
        
        # Processing time by level
        df.boxplot(column='processing_time', by='level', ax=axes[1])
        axes[1].set_title('Processing Time by Difficulty Level')
        axes[1].set_xlabel('Difficulty Level')
        axes[1].set_ylabel('Processing Time (seconds)')
        
        # Correlation between processing time and accuracy
        axes[2].scatter(df['processing_time'], df['final_correct'], alpha=0.6)
        axes[2].set_xlabel('Processing Time (seconds)')
        axes[2].set_ylabel('Final Answer Accuracy')
        axes[2].set_title('Processing Time vs Accuracy')
        
        # Add correlation coefficient
        corr = df['processing_time'].corr(df['final_correct'])
        axes[2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=axes[2].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle('Processing Time Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plot_file = output_dir / 'processing_time_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        # 4. Intent Classification and Retrieval Analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Intent classification accuracy by dataset
        intent_by_dataset = df.groupby('dataset')['intent_correct'].mean()
        intent_by_dataset.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Intent Classification Accuracy by Dataset')
        axes[0,0].set_ylabel('Accuracy Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Retrieval success rate by level
        retrieval_by_level = df.groupby('level')['is_relevant'].mean()
        retrieval_by_level.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Document Retrieval Success Rate by Level')
        axes[0,1].set_ylabel('Success Rate')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Relevance score vs accuracy
        axes[1,0].scatter(df['max_relevance_score'], df['final_correct'], alpha=0.6)
        axes[1,0].set_xlabel('Max Relevance Score')
        axes[1,0].set_ylabel('Final Answer Accuracy')
        axes[1,0].set_title('Document Relevance vs Answer Accuracy')
        
        # Add correlation
        corr = df['max_relevance_score'].corr(df['final_correct'])
        axes[1,0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[1,0].transAxes, fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Number of retrieved documents distribution
        doc_counts = df['num_retrieved_docs'].value_counts().sort_index()
        doc_counts.plot(kind='bar', ax=axes[1,1], color='orange')
        axes[1,1].set_title('Distribution of Retrieved Documents')
        axes[1,1].set_xlabel('Number of Retrieved Documents')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].tick_params(axis='x', rotation=0)
        
        plt.suptitle('Intent Classification and Document Retrieval Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plot_file = output_dir / 'intent_retrieval_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        # 5. Detailed Accuracy Breakdown
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create a detailed breakdown showing both accuracy types
        breakdown_data = []
        for dataset in df['dataset'].unique():
            for level in df['level'].unique():
                subset = df[(df['dataset'] == dataset) & (df['level'] == level)]
                if len(subset) > 0:
                    breakdown_data.append({
                        'Dataset_Level': f"{dataset}\n{level}",
                        'Summarized': subset['summarized_correct'].mean(),
                        'Final': subset['final_correct'].mean(),
                        'Count': len(subset)
                    })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        
        x = np.arange(len(breakdown_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, breakdown_df['Summarized'], width, 
                      label='Summarized Answer', alpha=0.8)
        bars2 = ax.bar(x + width/2, breakdown_df['Final'], width, 
                      label='Final Answer', alpha=0.8)
        
        ax.set_xlabel('Dataset and Difficulty Level')
        ax.set_ylabel('Accuracy Rate')
        ax.set_title('Detailed Accuracy Breakdown by Dataset and Level')
        ax.set_xticks(x)
        ax.set_xticklabels(breakdown_df['Dataset_Level'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, breakdown_df['Count'])):
            height = max(bar1.get_height(), bar2.get_height())
            ax.text(i, height + 0.01, f'n={count}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plot_file = output_dir / 'detailed_accuracy_breakdown.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        return plot_files
    
    def create_detailed_analysis(self, metrics: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create detailed question-by-question analysis"""
        df = pd.DataFrame(metrics)
        
        # Select relevant columns for detailed view
        detailed_df = df[[
            'dataset', 'level', 'question', 'expected_answer', 
            'summarized_answer', 'final_answer', 'summarized_correct', 
            'final_correct', 'intent_correct', 'processing_time', 
            'max_relevance_score', 'is_relevant'
        ]].copy()
        
        # Truncate long answers for readability
        for col in ['question', 'expected_answer', 'summarized_answer', 'final_answer']:
            detailed_df[col] = detailed_df[col].astype(str).str[:100] + "..."
        
        return detailed_df
    
    def save_analysis(self, tables: Dict[str, pd.DataFrame], detailed_df: pd.DataFrame, 
                     plot_files: List[str], output_file: str):
        """Save analysis to Excel file with plots"""
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Save summary tables
            for sheet_name, table in tables.items():
                table.to_excel(writer, sheet_name=sheet_name)
            
            # Save detailed analysis
            detailed_df.to_excel(writer, sheet_name="Detailed Analysis", index=False)
            
            # Create a plots reference sheet
            plots_df = pd.DataFrame({
                'Plot File': [Path(f).name for f in plot_files],
                'Full Path': plot_files,
                'Description': [
                    'Overview of accuracy, processing time, and relevance distributions',
                    'Heatmap showing accuracy by dataset and difficulty level',
                    'Analysis of processing times and correlation with accuracy',
                    'Intent classification and document retrieval performance',
                    'Detailed accuracy breakdown by dataset and level'
                ]
            })
            plots_df.to_excel(writer, sheet_name="Plot Files", index=False)
        
        print(f"Analysis saved to: {output_file}")
        print(f"Plot files created: {len(plot_files)} plots")
        for plot_file in plot_files:
            print(f"  - {plot_file}")
    
    def print_summary(self, tables: Dict[str, pd.DataFrame]):
        """Print summary to console"""
        print("\n" + "="*80)
        print("AGENT EVALUATION ANALYSIS")
        print("="*80)
        
        for table_name, table in tables.items():
            print(f"\n{table_name}:")
            print("-" * len(table_name))
            print(table.to_string())
    
    def analyze(self, output_file: str = None, create_plots: bool = True):
        """Run complete analysis"""
        metrics = self.generate_metrics()
        tables = self.create_summary_tables(metrics)
        detailed_df = self.create_detailed_analysis(metrics)
        
        # Print summary
        self.print_summary(tables)
        
        plot_files = []
        if create_plots:
            try:
                output_dir = Path(output_file).parent if output_file else Path(self.results_file).parent
                plot_files = self.create_plots(metrics, str(output_dir))
                print(f"\nCreated {len(plot_files)} visualization plots")
            except Exception as e:
                print(f"Warning: Could not create plots: {e}")
                print("Analysis will continue without plots...")
        
        # Save to file if specified
        if output_file:
            self.save_analysis(tables, detailed_df, plot_files, output_file)
        
        return tables, detailed_df, plot_files


def main():
    parser = argparse.ArgumentParser(description="Analyze agent evaluation results")
    parser.add_argument("--results_file", required=True, help="Path to evaluation results JSON file")
    parser.add_argument("--qa_dataset", default="gpt_qa_datasets_de.json", help="Path to QA dataset JSON file")
    parser.add_argument("--output", help="Output Excel file path (optional)")
    parser.add_argument("--no-plots", action="store_true", help="Skip creating plots")
    
    args = parser.parse_args()
    
    # Get QA dataset path relative to script location if not absolute
    if not Path(args.qa_dataset).is_absolute():
        script_dir = Path(__file__).parent
        qa_dataset_path = script_dir / args.qa_dataset
    else:
        qa_dataset_path = Path(args.qa_dataset)
    
    # Default output file if not specified
    if not args.output:
        results_path = Path(args.results_file)
        output_file = results_path.parent / f"{results_path.stem}_analysis.xlsx"
    else:
        output_file = args.output
    
    # Run analysis
    analyzer = ResultsAnalyzer(args.results_file, str(qa_dataset_path))
    tables, detailed_df, plot_files = analyzer.analyze(str(output_file), create_plots=not args.no_plots)
    
    print(f"\nAnalysis complete! Results saved to: {output_file}")
    if plot_files:
        print(f"Created {len(plot_files)} plots in the same directory")


if __name__ == "__main__":
    main()

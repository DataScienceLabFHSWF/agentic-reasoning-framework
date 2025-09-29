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
    
    def compare_answers(self, expected: Any, actual: str, question_type: str = "general", match_type: str = 'near') -> bool:
        """Compare expected and actual answers with different matching strategies."""
        if actual is None:
            return False
        
        expected_norm = self.normalize_answer(str(expected))
        actual_norm = self.normalize_answer(actual)

        if match_type == 'exact':
            return expected_norm == actual_norm

        # 'near' match logic (the previous default)
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
        # Get all agent answers from the result payload
        reasoning_answer = result.get("agent_responses", {}).get("reasoning_answer", "")
        summarized_answer = result.get("agent_responses", {}).get("summarized_answer", "")
        final_answer = result.get("agent_responses", {}).get("final_answer", "")
        
        # Differentiated accuracy checks
        final_answer_exact_correct = self.compare_answers(expected_answer, final_answer, match_type='exact')
        final_answer_near_correct = self.compare_answers(expected_answer, final_answer, match_type='near')
        reasoning_contains_answer = self.compare_answers(expected_answer, reasoning_answer, match_type='near')
        summarized_contains_answer = self.compare_answers(expected_answer, summarized_answer, match_type='near')
        
        # Intent classification accuracy
        is_corpus_relevant = result.get("intent_classification", {}).get("is_corpus_relevant")
        intent_correct = is_corpus_relevant is True  # All questions should be corpus-relevant
        
        # Processing time
        processing_time = result.get("workflow_metadata", {}).get("processing_time_seconds", 0)
        
        # Retrieval metrics
        is_relevant = result.get("workflow_metadata", {}).get("is_relevant", False)
        max_relevance_score = result.get("workflow_metadata", {}).get("max_relevance_score", 0)
        num_retrieved_docs = result.get("workflow_metadata", {}).get("num_retrieved_docs", 0)
        
        # Enhanced workflow metrics
        num_tool_calls = result.get("workflow_metadata", {}).get("num_tool_calls", 0)
        num_follow_up_questions = result.get("workflow_metadata", {}).get("num_follow_up_questions", 0)
        num_additional_context = result.get("workflow_metadata", {}).get("num_additional_context", 0)
        total_tool_execution_time = result.get("workflow_metadata", {}).get("total_tool_execution_time", 0.0)
        
        # Tool call analysis
        tool_calls = result.get("detailed_workflow_data", {}).get("tool_calls", [])
        unique_tools_used = len(set(tc.get("tool_name", "unknown") for tc in tool_calls))
        successful_tool_calls = sum(1 for tc in tool_calls if tc.get("success", True))
        failed_tool_calls = len(tool_calls) - successful_tool_calls
        
        # Follow-up question analysis
        follow_ups = result.get("detailed_workflow_data", {}).get("follow_up_questions", [])
        unique_follow_up_generators = len(set(fq.get("generated_by", "unknown") for fq in follow_ups))
        
        # Additional context analysis
        additional_context = result.get("detailed_workflow_data", {}).get("additional_context", [])
        avg_context_relevance = np.mean([ac.get("relevance_score", 0.0) for ac in additional_context]) if additional_context else 0.0
        total_context_length = sum(ac.get("content_length", 0) for ac in additional_context)
        
        return {
            "final_answer_exact_correct": final_answer_exact_correct,
            "final_answer_near_correct": final_answer_near_correct,
            "reasoning_contains_answer": reasoning_contains_answer,
            "summarized_contains_answer": summarized_contains_answer,
            "intent_correct": intent_correct,
            "processing_time": processing_time,
            "is_relevant": is_relevant,
            "max_relevance_score": max_relevance_score,
            "num_retrieved_docs": num_retrieved_docs,
            # Enhanced workflow metrics
            "num_tool_calls": num_tool_calls,
            "num_follow_up_questions": num_follow_up_questions,
            "num_additional_context": num_additional_context,
            "total_tool_execution_time": total_tool_execution_time,
            "tool_execution_percentage": (total_tool_execution_time / processing_time * 100) if processing_time > 0 else 0,
            "unique_tools_used": unique_tools_used,
            "successful_tool_calls": successful_tool_calls,
            "failed_tool_calls": failed_tool_calls,
            "tool_success_rate": (successful_tool_calls / num_tool_calls * 100) if num_tool_calls > 0 else 100,
            "unique_follow_up_generators": unique_follow_up_generators,
            "avg_context_relevance": avg_context_relevance,
            "total_context_length": total_context_length,
            "expected_answer": str(expected_answer),
            "reasoning_answer": reasoning_answer,
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
        """Create summary tables with micro and macro averages"""
        df = pd.DataFrame(metrics)
        
        tables = {}
        
        # Check if enhanced metrics are available
        has_enhanced_metrics = 'num_tool_calls' in df.columns and df['num_tool_calls'].notna().any()
        
        # Overall summary (Micro-average)
        overall_summary = {
            "Total Questions": len(df),
            "Final Answer Exact Acc.": f"{df['final_answer_exact_correct'].mean():.2%}",
            "Final Answer Near Acc.": f"{df['final_answer_near_correct'].mean():.2%}",
            "Reasoning Containment Acc.": f"{df['reasoning_contains_answer'].mean():.2%}",
            "Summarized Containment Acc.": f"{df['summarized_contains_answer'].mean():.2%}",
            "Intent Classification Acc.": f"{df['intent_correct'].mean():.2%}",
            "Avg Processing Time (s)": f"{df['processing_time'].mean():.2f}",
        }
        
        # Add enhanced metrics if available
        if has_enhanced_metrics:
            overall_summary.update({
                "Avg Tool Calls": f"{df['num_tool_calls'].mean():.1f}",
                "Avg Follow-ups": f"{df['num_follow_up_questions'].mean():.1f}",
                "Avg Additional Context": f"{df['num_additional_context'].mean():.1f}",
                "Tool Success Rate": f"{df['tool_success_rate'].mean():.1f}%",
                "Tool Execution Time %": f"{df['tool_execution_percentage'].mean():.1f}%"
            })
        
        tables["Overall Summary"] = pd.DataFrame([overall_summary])
        
        # --- Micro vs. Macro Accuracy Calculation ---
        def calculate_accuracies(grouped_df, group_col):
            # Micro: total correct / total questions
            micro_acc = grouped_df['final_answer_near_correct'].sum() / grouped_df['final_answer_near_correct'].count()
            
            # Macro: mean of per-group accuracies
            per_group_acc = df.groupby(group_col)['final_answer_near_correct'].mean()
            macro_acc = per_group_acc.mean()
            
            return pd.Series({'Micro Accuracy': micro_acc, 'Macro Accuracy': macro_acc})

        # By Dataset
        dataset_summary = df.groupby('dataset').agg({
            'final_answer_near_correct': ['count', 'mean'],
            'final_answer_exact_correct': 'mean',
            'reasoning_contains_answer': 'mean',
            'processing_time': 'mean',
        }).round(3)
        dataset_summary.columns = [
            'Total Questions', 'Final Near Acc.', 'Final Exact Acc.', 
            'Reasoning Containment', 'Avg Processing Time'
        ]
        
        dataset_acc_type = calculate_accuracies(df, 'dataset')
        dataset_summary['Micro Accuracy'] = dataset_acc_type['Micro Accuracy']
        dataset_summary['Macro Accuracy'] = dataset_acc_type['Macro Accuracy']
        tables["By Dataset"] = dataset_summary

        # By Level
        level_summary = df.groupby('level').agg({
            'final_answer_near_correct': ['count', 'mean'],
            'final_answer_exact_correct': 'mean',
            'reasoning_contains_answer': 'mean',
            'processing_time': 'mean',
        }).round(3)
        level_summary.columns = [
            'Total Questions', 'Final Near Acc.', 'Final Exact Acc.', 
            'Reasoning Containment', 'Avg Processing Time'
        ]
        
        level_acc_type = calculate_accuracies(df, 'level')
        level_summary['Micro Accuracy'] = level_acc_type['Micro Accuracy']
        level_summary['Macro Accuracy'] = level_acc_type['Macro Accuracy']
        tables["By Level"] = level_summary
        
        # By Dataset and Level
        dataset_level_summary = df.groupby(['dataset', 'level']).agg({
            'final_answer_near_correct': ['count', 'mean'],
            'final_answer_exact_correct': 'mean',
            'reasoning_contains_answer': 'mean',
            'processing_time': 'mean'
        }).round(3)
        
        dataset_level_summary.columns = [
            'Total Questions', 'Final Near Acc.', 'Final Exact Acc.', 
            'Reasoning Containment', 'Avg Processing Time'
        ]
        tables["By Dataset and Level"] = dataset_level_summary
        
        # Enhanced workflow summary (if metrics available)
        if has_enhanced_metrics:
            workflow_summary = df.groupby('dataset').agg({
                'num_tool_calls': ['count', 'mean', 'sum'],
                'num_follow_up_questions': ['mean', 'sum'],
                'num_additional_context': ['mean', 'sum'],
                'tool_success_rate': 'mean',
                'tool_execution_percentage': 'mean',
                'unique_tools_used': 'mean',
                'avg_context_relevance': 'mean'
            }).round(3)
            
            workflow_summary.columns = [
                'Questions', 'Avg Tool Calls', 'Total Tool Calls',
                'Avg Follow-ups', 'Total Follow-ups',
                'Avg Additional Context', 'Total Additional Context',
                'Tool Success Rate %', 'Tool Execution Time %',
                'Avg Unique Tools', 'Avg Context Relevance'
            ]
            tables["Enhanced Workflow Metrics"] = workflow_summary
        
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
        dataset_acc = df.groupby('dataset')[['final_answer_exact_correct', 'final_answer_near_correct']].mean()
        dataset_acc.plot(kind='bar', ax=axes[0,0], title='Accuracy by Dataset')
        axes[0,0].set_ylabel('Accuracy Rate')
        axes[0,0].set_xlabel('Dataset')
        axes[0,0].legend(['Final Exact Match', 'Final Near Match'])
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Accuracy by Level
        level_acc = df.groupby('level')[['final_answer_exact_correct', 'final_answer_near_correct']].mean()
        level_acc.plot(kind='bar', ax=axes[0,1], title='Accuracy by Difficulty Level')
        axes[0,1].set_ylabel('Accuracy Rate')
        axes[0,1].set_xlabel('Difficulty Level')
        axes[0,1].legend(['Final Exact Match', 'Final Near Match'])
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
        
        pivot_data = df.pivot_table(values='final_answer_near_correct', index='level', 
                                   columns='dataset', aggfunc='mean')
        
        sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0.5, 
                   ax=ax, fmt='.2f', cbar_kws={'label': 'Near Match Accuracy Rate'})
        ax.set_title('Final Answer Near Match Accuracy: Dataset vs Difficulty Level', 
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
        axes[2].scatter(df['processing_time'], df['final_answer_near_correct'], alpha=0.6)
        axes[2].set_xlabel('Processing Time (seconds)')
        axes[2].set_ylabel('Final Answer Near Accuracy')
        axes[2].set_title('Processing Time vs Accuracy')
        
        # Add correlation coefficient
        corr = df['processing_time'].corr(df['final_answer_near_correct'])
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
        axes[1,0].scatter(df['max_relevance_score'], df['final_answer_near_correct'], alpha=0.6)
        axes[1,0].set_xlabel('Max Relevance Score')
        axes[1,0].set_ylabel('Final Answer Near Accuracy')
        axes[1,0].set_title('Document Relevance vs Answer Accuracy')
        
        # Add correlation
        corr = df['max_relevance_score'].corr(df['final_answer_near_correct'])
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
                        'Exact Match': subset['final_answer_exact_correct'].mean(),
                        'Near Match': subset['final_answer_near_correct'].mean(),
                        'Count': len(subset)
                    })
        
        breakdown_df = pd.DataFrame(breakdown_data)
        
        x = np.arange(len(breakdown_df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, breakdown_df['Exact Match'], width, 
                      label='Final Exact Match', alpha=0.8)
        bars2 = ax.bar(x + width/2, breakdown_df['Near Match'], width, 
                      label='Final Near Match', alpha=0.8)
        
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
        
        # Check if enhanced metrics are available
        has_enhanced_metrics = 'num_tool_calls' in df.columns and df['num_tool_calls'].notna().any()
        
        if has_enhanced_metrics:
            # 6. Enhanced Workflow Analysis
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Tool calls distribution
            axes[0,0].hist(df['num_tool_calls'], bins=max(1, int(df['num_tool_calls'].max())), alpha=0.7, edgecolor='black')
            axes[0,0].set_title('Tool Calls Distribution')
            axes[0,0].set_xlabel('Number of Tool Calls')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].axvline(df['num_tool_calls'].mean(), color='red', linestyle='--',
                             label=f'Mean: {df["num_tool_calls"].mean():.1f}')
            axes[0,0].legend()
            
            # Follow-up questions distribution
            axes[0,1].hist(df['num_follow_up_questions'], bins=max(1, int(df['num_follow_up_questions'].max())), alpha=0.7, edgecolor='black')
            axes[0,1].set_title('Follow-up Questions Distribution')
            axes[0,1].set_xlabel('Number of Follow-up Questions')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].axvline(df['num_follow_up_questions'].mean(), color='red', linestyle='--',
                             label=f'Mean: {df["num_follow_up_questions"].mean():.1f}')
            axes[0,1].legend()
            
            # Additional context distribution
            axes[0,2].hist(df['num_additional_context'], bins=max(1, int(df['num_additional_context'].max())), alpha=0.7, edgecolor='black')
            axes[0,2].set_title('Additional Context Distribution')
            axes[0,2].set_xlabel('Number of Additional Context Items')
            axes[0,2].set_ylabel('Frequency')
            axes[0,2].axvline(df['num_additional_context'].mean(), color='red', linestyle='--',
                             label=f'Mean: {df["num_additional_context"].mean():.1f}')
            axes[0,2].legend()
            
            # Tool calls vs accuracy
            axes[1,0].scatter(df['num_tool_calls'], df['final_answer_near_correct'], alpha=0.6)
            axes[1,0].set_xlabel('Number of Tool Calls')
            axes[1,0].set_ylabel('Final Answer Near Accuracy')
            axes[1,0].set_title('Tool Calls vs Answer Accuracy')
            corr = df['num_tool_calls'].corr(df['final_answer_near_correct'])
            axes[1,0].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                          transform=axes[1,0].transAxes, fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Tool execution time percentage
            axes[1,1].hist(df['tool_execution_percentage'], bins=20, alpha=0.7, edgecolor='black')
            axes[1,1].set_title('Tool Execution Time as % of Total')
            axes[1,1].set_xlabel('Tool Execution Time (%)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].axvline(df['tool_execution_percentage'].mean(), color='red', linestyle='--',
                             label=f'Mean: {df["tool_execution_percentage"].mean():.1f}%')
            axes[1,1].legend()
            
            # Tool success rate
            axes[1,2].hist(df['tool_success_rate'], bins=10, alpha=0.7, edgecolor='black')
            axes[1,2].set_title('Tool Success Rate Distribution')
            axes[1,2].set_xlabel('Tool Success Rate (%)')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].axvline(df['tool_success_rate'].mean(), color='red', linestyle='--',
                             label=f'Mean: {df["tool_success_rate"].mean():.1f}%')
            axes[1,2].legend()
            
            plt.suptitle('Enhanced Workflow Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plot_file = output_dir / 'enhanced_workflow_analysis.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append(str(plot_file))
            plt.close()
            
            # 7. Workflow Efficiency Heatmap
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Tool calls heatmap
            tool_pivot = df.pivot_table(values='num_tool_calls', index='level', 
                                       columns='dataset', aggfunc='mean')
            sns.heatmap(tool_pivot, annot=True, cmap='YlOrRd', 
                       ax=axes[0], fmt='.1f', cbar_kws={'label': 'Average Tool Calls'})
            axes[0].set_title('Average Tool Calls: Dataset vs Difficulty Level')
            
            # Follow-up questions heatmap
            followup_pivot = df.pivot_table(values='num_follow_up_questions', index='level', 
                                           columns='dataset', aggfunc='mean')
            sns.heatmap(followup_pivot, annot=True, cmap='Blues', 
                       ax=axes[1], fmt='.1f', cbar_kws={'label': 'Average Follow-up Questions'})
            axes[1].set_title('Average Follow-up Questions: Dataset vs Difficulty Level')
            
            plt.tight_layout()
            plot_file = output_dir / 'workflow_efficiency_heatmap.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append(str(plot_file))
            plt.close()
        
        return plot_files
    
    def create_detailed_analysis(self, metrics: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create detailed question-by-question analysis"""
        df = pd.DataFrame(metrics)
        
        # Check if enhanced metrics are available
        has_enhanced_metrics = 'num_tool_calls' in df.columns and df['num_tool_calls'].notna().any()
        
        # Select relevant columns for detailed view
        detailed_columns = [
            'dataset', 'level', 'question', 'expected_answer', 
            'final_answer', 'reasoning_answer', 'summarized_answer',
            'final_answer_exact_correct', 'final_answer_near_correct',
            'reasoning_contains_answer', 'summarized_contains_answer',
            'intent_correct', 'processing_time', 
            'max_relevance_score', 'is_relevant'
        ]
        
        # Add enhanced columns if available
        if has_enhanced_metrics:
            detailed_columns.extend([
                'num_tool_calls', 'num_follow_up_questions', 'num_additional_context',
                'tool_success_rate', 'tool_execution_percentage', 'unique_tools_used',
                'avg_context_relevance', 'total_context_length'
            ])
        
        detailed_df = df[detailed_columns].copy()
        
        # Truncate long answers for readability
        for col in ['question', 'expected_answer', 'final_answer', 'reasoning_answer', 'summarized_answer']:
            if col in detailed_df.columns:
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
    
    def analyze(self, output_file: str = None, create_plots: bool = True, output_folder: str = None):
        """Run complete analysis"""
        metrics = self.generate_metrics()
        
        if not metrics:
            print("No valid results found to analyze.")
            return None, None, None

        tables = self.create_summary_tables(metrics)
        detailed_df = self.create_detailed_analysis(metrics)
        
        # Determine output directory
        if output_folder:
            output_dir = Path(output_folder)
        elif output_file:
            output_dir = Path(output_file).parent
        else:
            output_dir = Path(self.results_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine output file path
        if output_file:
            output_filepath = Path(output_file)
            if not output_filepath.is_absolute():
                 output_filepath = output_dir / output_filepath.name
        else:
            results_path = Path(self.results_file)
            output_filepath = output_dir / f"{results_path.stem}_analysis.xlsx"

        # Print summary
        self.print_summary(tables)
        
        plot_files = []
        if create_plots:
            try:
                plot_files = self.create_plots(metrics, str(output_dir))
                print(f"\nCreated {len(plot_files)} visualization plots in {output_dir}")
            except Exception as e:
                print(f"Warning: Could not create plots: {e}")
                print("Analysis will continue without plots...")
        
        # Save to file if specified
        self.save_analysis(tables, detailed_df, plot_files, str(output_filepath))
        
        return tables, detailed_df, plot_files


def main():
    parser = argparse.ArgumentParser(description="Analyze agent evaluation results")
    parser.add_argument("--results_file", required=True, help="Path to evaluation results JSON file")
    parser.add_argument("--qa_dataset", default="gpt_qa_datasets_de.json", help="Path to QA dataset JSON file")
    parser.add_argument("--output", help="Output Excel file name (e.g., analysis.xlsx). Saved in --output_folder.")
    parser.add_argument("--output_folder", help="Folder to save all analysis results (Excel, plots). Defaults to results_file directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip creating plots")
    
    args = parser.parse_args()
    
    # Get QA dataset path relative to script location if not absolute
    if not Path(args.qa_dataset).is_absolute():
        script_dir = Path(__file__).parent
        qa_dataset_path = script_dir / args.qa_dataset
    else:
        qa_dataset_path = Path(args.qa_dataset)
    
    # Run analysis
    analyzer = ResultsAnalyzer(args.results_file, str(qa_dataset_path))
    tables, detailed_df, plot_files = analyzer.analyze(
        output_file=args.output, 
        create_plots=not args.no_plots,
        output_folder=args.output_folder
    )
    
    if tables:
        print(f"\nAnalysis complete! Results saved.")


if __name__ == "__main__":
    main()

"""
compare_models.py
Script to compare GPT, Gemini, and Reasoning Agent answers against ground truth
and analyze consistency between models
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re
from itertools import combinations
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """Compare multiple models' answers against ground truth and each other"""
    
    def __init__(self, qa_dataset_path: str, output_dir: str = None):
        self.qa_dataset_path = qa_dataset_path
        self.output_dir = Path(output_dir) if output_dir else Path(qa_dataset_path).parent / "combined_evaluation"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load ground truth
        self.ground_truth = self.load_ground_truth()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_ground_truth(self) -> Dict[str, List[Dict]]:
        """Load ground truth QA dataset"""
        with open(self.qa_dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_model_answers(self, file_path: str) -> Dict[str, List[Dict]]:
        """Load model answers from JSON file"""
        if not Path(file_path).exists():
            print(f"Warning: File not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different file formats
        if 'evaluation_results' in data:
            # This is a reasoning agent results file
            formatted_data = {}
            for dataset_name, dataset_results in data['evaluation_results'].items():
                if 'results' in dataset_results:
                    formatted_data[dataset_name] = []
                    for result in dataset_results['results']:
                        formatted_data[dataset_name].append({
                            'question': result.get('question', ''),
                            'answer': result.get('agent_responses', {}).get('final_answer', '')
                        })
            return formatted_data
        else:
            # This is GPT/Gemini format - need to convert to consistent format
            formatted_data = {}
            for dataset_name, questions in data.items():
                if isinstance(questions, list) and questions:  # Skip empty datasets
                    formatted_data[dataset_name] = []
                    for q_data in questions:
                        # Handle both 'gpt_answer' and 'answer' field names
                        answer = q_data.get('gpt_answer') or q_data.get('answer', '')
                        formatted_data[dataset_name].append({
                            'question': q_data.get('question', ''),
                            'answer': answer
                        })
            return formatted_data
    
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
    
    def compare_answers(self, expected: Any, actual: str, match_type: str = 'near') -> bool:
        """Compare expected and actual answers with different matching strategies"""
        if actual is None or actual == "":
            return False
        
        expected_norm = self.normalize_answer(str(expected))
        actual_norm = self.normalize_answer(actual)

        if match_type == 'exact':
            return expected_norm == actual_norm

        # 'near' match logic
        # Check for exact match
        if expected_norm == actual_norm:
            return True
        
        # For numeric answers
        expected_num = self.extract_numeric_value(expected)
        actual_num = self.extract_numeric_value(actual)
        if expected_num is not None and actual_num is not None:
            # Allow small tolerance for numeric values
            return abs(expected_num - actual_num) < 0.01
        
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
    
    def calculate_consistency(self, answers: Dict[str, str]) -> Dict[str, bool]:
        """Calculate pairwise consistency between models"""
        models = list(answers.keys())
        consistency = {}
        
        for model1, model2 in combinations(models, 2):
            pair_key = f"{model1}_vs_{model2}"
            # Use near match for consistency
            consistency[pair_key] = self.compare_answers(answers[model1], answers[model2], 'near')
        
        return consistency
    
    def create_combined_dataset(self, model_files: Dict[str, str]) -> pd.DataFrame:
        """Create combined dataset with all models' answers"""
        combined_data = []
        
        # Load all model answers
        model_answers = {}
        for model_name, file_path in model_files.items():
            model_answers[model_name] = self.load_model_answers(file_path)
            print(f"Loaded {model_name} with datasets: {list(model_answers[model_name].keys())}")
        
        # Process each dataset
        for dataset_name, questions in self.ground_truth.items():
            # Skip dataset4 for now as it's not consistently available in all models
            if dataset_name == 'dataset4':
                continue
                
            print(f"\nProcessing {dataset_name} with {len(questions)} questions")
            
            for i, question_data in enumerate(questions):
                question = question_data['question']
                ground_truth_answer = question_data['answer']
                level = question_data['level']
                
                # Collect answers from all models
                row = {
                    'dataset': dataset_name,
                    'question_index': i,
                    'question': question,
                    'level': level,
                    'ground_truth': ground_truth_answer
                }
                
                # Add model answers
                model_answers_dict = {}
                for model_name in model_files.keys():
                    answer = ""
                    if (dataset_name in model_answers[model_name] and 
                        i < len(model_answers[model_name][dataset_name])):
                        answer_data = model_answers[model_name][dataset_name][i]
                        answer = answer_data.get('answer', '')
                        
                        # Handle nested answer structures (for reasoning agent)
                        if isinstance(answer, dict):
                            answer = answer.get('final_answer', str(answer))
                    
                    row[f'{model_name}_answer'] = answer
                    model_answers_dict[model_name] = answer
                
                # Calculate accuracy for each model
                for model_name in model_files.keys():
                    model_answer = row[f'{model_name}_answer']
                    row[f'{model_name}_exact_correct'] = self.compare_answers(ground_truth_answer, model_answer, 'exact')
                    row[f'{model_name}_near_correct'] = self.compare_answers(ground_truth_answer, model_answer, 'near')
                
                # Calculate pairwise consistency
                consistency = self.calculate_consistency(model_answers_dict)
                for pair, is_consistent in consistency.items():
                    row[f'consistency_{pair}'] = is_consistent
                
                combined_data.append(row)
        
        return pd.DataFrame(combined_data)
    
    def calculate_summary_metrics(self, df: pd.DataFrame, model_names: List[str]) -> Dict[str, pd.DataFrame]:
        """Calculate comprehensive summary metrics"""
        summaries = {}
        
        # Overall summary (Micro-average)
        overall_data = []
        for model in model_names:
            exact_acc = df[f'{model}_exact_correct'].mean()
            near_acc = df[f'{model}_near_correct'].mean()
            total_questions = len(df)
            
            overall_data.append({
                'Model': model.upper(),
                'Total Questions': total_questions,
                'Exact Match Accuracy': f"{exact_acc:.2%}",
                'Near Match Accuracy': f"{near_acc:.2%}",
                'Exact Match Count': df[f'{model}_exact_correct'].sum(),
                'Near Match Count': df[f'{model}_near_correct'].sum()
            })
        
        summaries['Overall Performance'] = pd.DataFrame(overall_data)
        
        # By Dataset
        dataset_data = []
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            for model in model_names:
                exact_acc = dataset_df[f'{model}_exact_correct'].mean()
                near_acc = dataset_df[f'{model}_near_correct'].mean()
                
                dataset_data.append({
                    'Dataset': dataset,
                    'Model': model.upper(),
                    'Questions': len(dataset_df),
                    'Exact Accuracy': f"{exact_acc:.2%}",
                    'Near Accuracy': f"{near_acc:.2%}"
                })
        
        summaries['Performance by Dataset'] = pd.DataFrame(dataset_data)
        
        # By Level
        level_data = []
        for level in df['level'].unique():
            level_df = df[df['level'] == level]
            for model in model_names:
                exact_acc = level_df[f'{model}_exact_correct'].mean()
                near_acc = level_df[f'{model}_near_correct'].mean()
                
                level_data.append({
                    'Level': level,
                    'Model': model.upper(),
                    'Questions': len(level_df),
                    'Exact Accuracy': f"{exact_acc:.2%}",
                    'Near Accuracy': f"{near_acc:.2%}"
                })
        
        summaries['Performance by Level'] = pd.DataFrame(level_data)
        
        # Consistency Analysis
        consistency_cols = [col for col in df.columns if col.startswith('consistency_')]
        if consistency_cols:
            consistency_data = []
            for col in consistency_cols:
                pair_name = col.replace('consistency_', '').replace('_vs_', ' vs ')
                consistency_rate = df[col].mean()
                agreement_count = df[col].sum()
                
                consistency_data.append({
                    'Model Pair': pair_name.upper(),
                    'Consistency Rate': f"{consistency_rate:.2%}",
                    'Agreement Count': f"{agreement_count}/{len(df)}",
                    'Disagreement Count': len(df) - agreement_count
                })
            
            summaries['Model Consistency'] = pd.DataFrame(consistency_data)
        
        # Micro vs Macro Analysis
        micro_macro_data = []
        for model in model_names:
            # Micro accuracy (overall)
            micro_exact = df[f'{model}_exact_correct'].mean()
            micro_near = df[f'{model}_near_correct'].mean()
            
            # Macro accuracy (mean of dataset accuracies)
            dataset_exact_accs = df.groupby('dataset')[f'{model}_exact_correct'].mean()
            dataset_near_accs = df.groupby('dataset')[f'{model}_near_correct'].mean()
            macro_exact = dataset_exact_accs.mean()
            macro_near = dataset_near_accs.mean()
            
            micro_macro_data.append({
                'Model': model.upper(),
                'Micro Exact': f"{micro_exact:.2%}",
                'Macro Exact': f"{macro_exact:.2%}",
                'Micro Near': f"{micro_near:.2%}",
                'Macro Near': f"{macro_near:.2%}",
                'Exact Diff (Macro-Micro)': f"{(macro_exact - micro_exact):.2%}",
                'Near Diff (Macro-Micro)': f"{(macro_near - micro_near):.2%}"
            })
        
        summaries['Micro vs Macro'] = pd.DataFrame(micro_macro_data)
        
        return summaries
    
    def create_visualizations(self, df: pd.DataFrame, model_names: List[str]) -> List[str]:
        """Create comprehensive visualizations"""
        plot_files = []
        
        # Set up colors for models
        colors = sns.color_palette("husl", len(model_names))
        model_colors = dict(zip(model_names, colors))
        
        # 1. Overall Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy by model
        accuracy_data = []
        for model in model_names:
            exact_acc = df[f'{model}_exact_correct'].mean()
            near_acc = df[f'{model}_near_correct'].mean()
            accuracy_data.append({
                'Model': model.upper(),
                'Exact Match': exact_acc,
                'Near Match': near_acc
            })
        
        acc_df = pd.DataFrame(accuracy_data)
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, acc_df['Exact Match'], width, label='Exact Match', alpha=0.8)
        axes[0,0].bar(x + width/2, acc_df['Near Match'], width, label='Near Match', alpha=0.8)
        axes[0,0].set_xlabel('Model')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_title('Overall Accuracy by Model')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([m.upper() for m in model_names])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Performance by dataset
        dataset_performance = []
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            for model in model_names:
                near_acc = dataset_df[f'{model}_near_correct'].mean()
                dataset_performance.append({
                    'Dataset': dataset,
                    'Model': model.upper(),
                    'Accuracy': near_acc
                })
        
        perf_df = pd.DataFrame(dataset_performance)
        pivot_perf = perf_df.pivot(index='Dataset', columns='Model', values='Accuracy')
        pivot_perf.plot(kind='bar', ax=axes[0,1], width=0.8)
        axes[0,1].set_title('Near Match Accuracy by Dataset')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend(title='Model')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Performance by level
        level_performance = []
        for level in df['level'].unique():
            level_df = df[df['level'] == level]
            for model in model_names:
                near_acc = level_df[f'{model}_near_correct'].mean()
                level_performance.append({
                    'Level': level,
                    'Model': model.upper(),
                    'Accuracy': near_acc
                })
        
        level_df = pd.DataFrame(level_performance)
        pivot_level = level_df.pivot(index='Level', columns='Model', values='Accuracy')
        pivot_level.plot(kind='bar', ax=axes[1,0], width=0.8)
        axes[1,0].set_title('Near Match Accuracy by Difficulty Level')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].legend(title='Model')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Consistency heatmap
        consistency_cols = [col for col in df.columns if col.startswith('consistency_')]
        if consistency_cols:
            consistency_rates = []
            pair_labels = []
            for col in consistency_cols:
                rate = df[col].mean()
                pair_name = col.replace('consistency_', '').replace('_vs_', ' vs ')
                consistency_rates.append(rate)
                pair_labels.append(pair_name.upper())
            
            y_pos = np.arange(len(pair_labels))
            bars = axes[1,1].barh(y_pos, consistency_rates, alpha=0.8)
            axes[1,1].set_yticks(y_pos)
            axes[1,1].set_yticklabels(pair_labels)
            axes[1,1].set_xlabel('Consistency Rate')
            axes[1,1].set_title('Model Consistency (Agreement Rate)')
            axes[1,1].grid(True, alpha=0.3)
            
            # Add percentage labels on bars
            for i, (bar, rate) in enumerate(zip(bars, consistency_rates)):
                axes[1,1].text(rate + 0.01, bar.get_y() + bar.get_height()/2, 
                              f'{rate:.1%}', va='center', fontsize=10)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'model_comparison_overview.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        # 2. Detailed Heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy heatmap by dataset and model
        dataset_model_acc = []
        for dataset in df['dataset'].unique():
            for model in model_names:
                dataset_df = df[df['dataset'] == dataset]
                near_acc = dataset_df[f'{model}_near_correct'].mean()
                dataset_model_acc.append({
                    'Dataset': dataset,
                    'Model': model.upper(),
                    'Accuracy': near_acc
                })
        
        heatmap_df = pd.DataFrame(dataset_model_acc).pivot(index='Dataset', columns='Model', values='Accuracy')
        sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn', center=0.5, 
                   ax=axes[0], fmt='.2f', cbar_kws={'label': 'Near Match Accuracy'})
        axes[0].set_title('Accuracy by Dataset and Model')
        
        # Accuracy heatmap by level and model
        level_model_acc = []
        for level in df['level'].unique():
            for model in model_names:
                level_df = df[df['level'] == level]
                near_acc = level_df[f'{model}_near_correct'].mean()
                level_model_acc.append({
                    'Level': level,
                    'Model': model.upper(),
                    'Accuracy': near_acc
                })
        
        level_heatmap_df = pd.DataFrame(level_model_acc).pivot(index='Level', columns='Model', values='Accuracy')
        sns.heatmap(level_heatmap_df, annot=True, cmap='RdYlGn', center=0.5, 
                   ax=axes[1], fmt='.2f', cbar_kws={'label': 'Near Match Accuracy'})
        axes[1].set_title('Accuracy by Level and Model')
        
        # Model agreement matrix
        if len(model_names) >= 2:
            agreement_matrix = np.eye(len(model_names))
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i != j:
                        col_name = f'consistency_{model1}_vs_{model2}'
                        if col_name in df.columns:
                            agreement_matrix[i, j] = df[col_name].mean()
                        else:
                            col_name = f'consistency_{model2}_vs_{model1}'
                            if col_name in df.columns:
                                agreement_matrix[i, j] = df[col_name].mean()
            
            sns.heatmap(agreement_matrix, annot=True, cmap='Blues', 
                       xticklabels=[m.upper() for m in model_names],
                       yticklabels=[m.upper() for m in model_names],
                       ax=axes[2], fmt='.2f', cbar_kws={'label': 'Agreement Rate'})
            axes[2].set_title('Model Agreement Matrix')
        
        plt.tight_layout()
        plot_file = self.output_dir / 'detailed_heatmaps.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        # 3. Error Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error Analysis and Model Comparison', fontsize=16, fontweight='bold')
        
        # Error rates by dataset
        error_data = []
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]
            for model in model_names:
                error_rate = 1 - dataset_df[f'{model}_near_correct'].mean()
                error_data.append({
                    'Dataset': dataset,
                    'Model': model.upper(),
                    'Error_Rate': error_rate
                })
        
        error_df = pd.DataFrame(error_data)
        pivot_error = error_df.pivot(index='Dataset', columns='Model', values='Error_Rate')
        pivot_error.plot(kind='bar', ax=axes[0,0], width=0.8)
        axes[0,0].set_title('Error Rate by Dataset')
        axes[0,0].set_ylabel('Error Rate')
        axes[0,0].legend(title='Model')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Error rates by level
        error_level_data = []
        for level in df['level'].unique():
            level_df = df[df['level'] == level]
            for model in model_names:
                error_rate = 1 - level_df[f'{model}_near_correct'].mean()
                error_level_data.append({
                    'Level': level,
                    'Model': model.upper(),
                    'Error_Rate': error_rate
                })
        
        error_level_df = pd.DataFrame(error_level_data)
        pivot_error_level = error_level_df.pivot(index='Level', columns='Model', values='Error_Rate')
        pivot_error_level.plot(kind='bar', ax=axes[0,1], width=0.8)
        axes[0,1].set_title('Error Rate by Difficulty Level')
        axes[0,1].set_ylabel('Error Rate')
        axes[0,1].legend(title='Model')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Scatter plot of model correlations (if we have at least 2 models)
        if len(model_names) >= 2:
            model1, model2 = model_names[0], model_names[1]
            x_acc = df[f'{model1}_near_correct'].astype(float)
            y_acc = df[f'{model2}_near_correct'].astype(float)
            
            axes[1,0].scatter(x_acc, y_acc, alpha=0.6, s=50)
            axes[1,0].set_xlabel(f'{model1.upper()} Accuracy (0=Wrong, 1=Correct)')
            axes[1,0].set_ylabel(f'{model2.upper()} Accuracy (0=Wrong, 1=Correct)')
            axes[1,0].set_title(f'{model1.upper()} vs {model2.upper()} Performance')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            correlation = np.corrcoef(x_acc, y_acc)[0, 1]
            axes[1,0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                          transform=axes[1,0].transAxes, fontsize=12,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Model ranking distribution
        if len(model_names) >= 2:
            ranking_data = []
            for idx, row in df.iterrows():
                scores = {}
                for model in model_names:
                    scores[model] = row[f'{model}_near_correct']
                
                # Rank models for this question (1 = best, higher = worse)
                sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (model, score) in enumerate(sorted_models, 1):
                    ranking_data.append({
                        'Model': model.upper(),
                        'Rank': rank,
                        'Question': idx
                    })
            
            ranking_df = pd.DataFrame(ranking_data)
            rank_counts = ranking_df.groupby(['Model', 'Rank']).size().unstack(fill_value=0)
            
            rank_counts.plot(kind='bar', stacked=True, ax=axes[1,1], 
                           colormap='RdYlBu_r', width=0.8)
            axes[1,1].set_title('Model Ranking Distribution')
            axes[1,1].set_ylabel('Number of Questions')
            axes[1,1].set_xlabel('Model')
            axes[1,1].legend(title='Rank', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'error_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(str(plot_file))
        plt.close()
        
        return plot_files
    
    def save_results(self, df: pd.DataFrame, summaries: Dict[str, pd.DataFrame], 
                    plot_files: List[str], model_names: List[str]):
        """Save all results to files"""
        
        # Save detailed combined dataset
        detailed_file = self.output_dir / 'combined_detailed_results.xlsx'
        df.to_excel(detailed_file, index=False)
        
        # Save summary metrics
        summary_file = self.output_dir / 'model_comparison_summary.xlsx'
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            for sheet_name, summary_df in summaries.items():
                summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Add plot references
            plots_df = pd.DataFrame({
                'Plot File': [Path(f).name for f in plot_files],
                'Full Path': plot_files,
                'Description': [
                    'Overview comparison of all models including consistency analysis',
                    'Detailed heatmaps showing performance patterns',
                    'Error analysis and model ranking distributions'
                ]
            })
            plots_df.to_excel(writer, sheet_name="Plot Files", index=False)
        
        # Save JSON summary for easy programmatic access
        json_summary = {
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'models_compared': model_names,
                'total_questions': len(df),
                'datasets': df['dataset'].unique().tolist(),
                'difficulty_levels': df['level'].unique().tolist()
            },
            'overall_performance': {},
            'consistency_analysis': {}
        }
        
        # Add overall performance to JSON
        for model in model_names:
            exact_acc = df[f'{model}_exact_correct'].mean()
            near_acc = df[f'{model}_near_correct'].mean()
            json_summary['overall_performance'][model] = {
                'exact_match_accuracy': round(exact_acc, 4),
                'near_match_accuracy': round(near_acc, 4),
                'exact_match_count': int(df[f'{model}_exact_correct'].sum()),
                'near_match_count': int(df[f'{model}_near_correct'].sum())
            }
        
        # Add consistency analysis
        consistency_cols = [col for col in df.columns if col.startswith('consistency_')]
        for col in consistency_cols:
            pair_name = col.replace('consistency_', '')
            consistency_rate = df[col].mean()
            agreement_count = int(df[col].sum())
            json_summary['consistency_analysis'][pair_name] = {
                'consistency_rate': round(consistency_rate, 4),
                'agreement_count': agreement_count,
                'disagreement_count': len(df) - agreement_count
            }
        
        json_file = self.output_dir / 'model_comparison_summary.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Results saved to: {self.output_dir}")
        print(f"📊 Detailed results: {detailed_file}")
        print(f"📈 Summary metrics: {summary_file}")
        print(f"🔗 JSON summary: {json_file}")
        print(f"📁 Plots created: {len(plot_files)} visualization files")
    
    def print_summary(self, summaries: Dict[str, pd.DataFrame], model_names: List[str]):
        """Print summary to console"""
        print("\n" + "="*80)
        print("MODEL COMPARISON ANALYSIS")
        print("="*80)
        
        for table_name, table in summaries.items():
            print(f"\n{table_name}:")
            print("-" * len(table_name))
            print(table.to_string(index=False))
    
    def run_comparison(self, model_files: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
        """Run complete model comparison"""
        print(f"Starting model comparison...")
        print(f"Models to compare: {', '.join(model_files.keys())}")
        print(f"Output directory: {self.output_dir}")
        
        # Create combined dataset
        print("\nCreating combined dataset...")
        df = self.create_combined_dataset(model_files)
        print(f"Combined dataset created with {len(df)} questions")
        
        # Calculate metrics
        print("Calculating summary metrics...")
        model_names = list(model_files.keys())
        summaries = self.calculate_summary_metrics(df, model_names)
        
        # Create visualizations
        print("Creating visualizations...")
        plot_files = self.create_visualizations(df, model_names)
        
        # Print summary
        self.print_summary(summaries, model_names)
        
        # Save results
        print("\nSaving results...")
        self.save_results(df, summaries, plot_files, model_names)
        
        return df, summaries, plot_files


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Compare multiple models against ground truth")
    parser.add_argument("--qa_dataset", default="gpt_qa_datasets_de.json", 
                       help="Path to ground truth QA dataset")
    parser.add_argument("--gpt_answers", default="gpt_ans.json", 
                       help="Path to GPT answers file")
    parser.add_argument("--gemini_answers", default="gemini_ans.json", 
                       help="Path to Gemini answers file")
    parser.add_argument("--reasoning_agent_answers", default="my_simple_reasoning_agent_ans.json", 
                       help="Path to reasoning agent answers file")
    parser.add_argument("--output_dir", default="combined_evaluation", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Get absolute paths
    script_dir = Path(__file__).parent
    
    qa_dataset_path = script_dir / args.qa_dataset
    if not qa_dataset_path.exists():
        raise FileNotFoundError(f"Ground truth dataset not found: {qa_dataset_path}")
    
    # Define model files to compare
    model_files = {
        'gpt': str(script_dir / args.gpt_answers),
        'gemini': str(script_dir / args.gemini_answers),
        'reasoning_agent': str(script_dir / args.reasoning_agent_answers)
    }
    
    # Check which files exist
    existing_models = {}
    for model_name, file_path in model_files.items():
        if Path(file_path).exists():
            existing_models[model_name] = file_path
            print(f"✅ Found {model_name} answers: {file_path}")
        else:
            print(f"❌ Missing {model_name} answers: {file_path}")
    
    if len(existing_models) < 2:
        raise ValueError("Need at least 2 model answer files to run comparison")
    
    # Run comparison
    comparator = ModelComparator(str(qa_dataset_path), args.output_dir)
    df, summaries, plot_files = comparator.run_comparison(existing_models)
    
    print(f"\n🎉 Model comparison complete!")
    print(f"📁 All results saved in: {comparator.output_dir}")


if __name__ == "__main__":
    main()

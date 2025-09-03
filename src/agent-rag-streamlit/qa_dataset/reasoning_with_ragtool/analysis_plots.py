import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for extra high-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 600  # Increased from 300
plt.rcParams['savefig.dpi'] = 600  # Increased from 300
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'none'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.8

def load_and_prepare_data(csv_path):
    """Load and prepare the dataset for analysis."""
    df = pd.read_csv(csv_path)
    
    # Convert boolean columns to numeric - handle string values properly
    bool_columns = ['Exact Match', 'Near Match', 'Semantic Match']
    for col in bool_columns:
        # Convert string values to boolean, then to int
        df[col] = df[col].astype(str).str.upper()  # Ensure uppercase
        df[col] = df[col].map({'TRUE': 1, 'FALSE': 0})
        # Handle any NaN values that might result from unexpected strings
        df[col] = df[col].fillna(0).astype(int)
    
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample of boolean columns:")
    print(df[bool_columns].head())
    
    return df

def plot_match_types_by_difficulty(df, save_path):
    """Create bar plot of match types by difficulty level showing percentages."""
    # Prepare data for plotting
    match_data = []
    difficulties = sorted(df['Difficulty'].unique())
    
    print(f"Difficulties found: {difficulties}")
    
    for difficulty in difficulties:
        subset = df[df['Difficulty'] == difficulty]
        total_questions = len(subset)
        
        if total_questions > 0:
            exact_pct = (subset['Exact Match'].sum() / total_questions) * 100
            near_pct = (subset['Near Match'].sum() / total_questions) * 100
            semantic_pct = (subset['Semantic Match'].sum() / total_questions) * 100
        else:
            exact_pct = near_pct = semantic_pct = 0
        
        print(f"Difficulty {difficulty}: Total={total_questions}, Exact={exact_pct:.1f}%, Near={near_pct:.1f}%, Semantic={semantic_pct:.1f}%")
        
        match_data.extend([
            {'Difficulty': difficulty, 'Match Type': 'Exact Match', 'Percentage': exact_pct},
            {'Difficulty': difficulty, 'Match Type': 'Near Match', 'Percentage': near_pct},
            {'Difficulty': difficulty, 'Match Type': 'Semantic Match', 'Percentage': semantic_pct}
        ])
    
    match_df = pd.DataFrame(match_data)
    print(f"Match data prepared: {match_df}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=match_df, x='Difficulty', y='Percentage', hue='Match Type')
    
    plt.title('Accuracy by Level', fontweight='bold', pad=20)
    plt.xlabel('Difficulty Level', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.legend(title='Match Type', title_fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add percentage labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'accuracy_by_level.png', dpi=600, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='png')
    plt.close()

def plot_processing_time_distribution(df, save_path):
    """Create distribution plot for processing times with median and average highlighted."""
    # Filter out outliers above 600 seconds
    df_filtered = df[df['Processing Time (s)'] <= 600]
    
    plt.figure(figsize=(12, 8))
    
    # Create the distribution plot
    sns.histplot(data=df_filtered, x='Processing Time (s)', kde=True, alpha=0.7, bins=20)
    
    # Calculate and highlight median and average
    median_time = df_filtered['Processing Time (s)'].median()
    mean_time = df_filtered['Processing Time (s)'].mean()
    
    # Add median line
    plt.axvline(median_time, color='red', linestyle='--', linewidth=2, label=f'Median: {median_time:.1f}s')
    
    # Add average line
    plt.axvline(mean_time, color='blue', linestyle='--', linewidth=2, label=f'Average: {mean_time:.1f}s')
    
    # Add median annotation
    plt.annotate(f'Median: {median_time:.1f}s', 
                xy=(median_time, plt.ylim()[1] * 0.8), 
                xytext=(median_time + 10, plt.ylim()[1] * 0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=12, fontweight='bold', color='red')
    
    # Add average annotation
    plt.annotate(f'Average: {mean_time:.1f}s', 
                xy=(mean_time, plt.ylim()[1] * 0.7), 
                xytext=(mean_time + 10, plt.ylim()[1] * 0.8),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=12, fontweight='bold', color='blue')
    
    plt.title('Distribution of Question Processing Times', fontweight='bold', pad=20)
    plt.xlabel('Processing Time (seconds)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.legend(frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'processing_time_distribution.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.close()

def plot_tool_calls_analysis(df, save_path):
    """Analyze tool calls distribution and relationship with difficulty."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Tool calls distribution
    tool_calls_counts = df['Tool Calls'].value_counts().sort_index()
    bars1 = ax1.bar(tool_calls_counts.index, tool_calls_counts.values, alpha=0.8)
    ax1.set_title('Distribution of Tool Calls per Question', fontweight='bold')
    ax1.set_xlabel('Number of Tool Calls', fontweight='bold')
    ax1.set_ylabel('Number of Questions', fontweight='bold')
    
    # Add value labels
    for i, v in enumerate(tool_calls_counts.values):
        ax1.text(tool_calls_counts.index[i], v + 0.1, str(v), ha='center', fontweight='bold')
    
    # Tool calls by difficulty
    sns.boxplot(data=df, x='Difficulty', y='Tool Calls', ax=ax2)
    ax2.set_title('Tool Calls Distribution by Difficulty Level', fontweight='bold')
    ax2.set_xlabel('Difficulty Level', fontweight='bold')
    ax2.set_ylabel('Number of Tool Calls', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path / 'tool_calls_analysis.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.close()

def plot_dataset_performance_comparison(df, save_path):
    """Compare performance across different datasets."""
    plt.figure(figsize=(14, 8))
    
    # Calculate success rates by dataset
    dataset_stats = []
    datasets = sorted(df['Dataset'].unique())
    
    for dataset in datasets:
        subset = df[df['Dataset'] == dataset]
        total_questions = len(subset)
        exact_rate = subset['Exact Match'].mean() * 100
        near_rate = subset['Near Match'].mean() * 100
        semantic_rate = subset['Semantic Match'].mean() * 100
        
        print(f"Dataset {dataset}: Total={total_questions}, Exact={exact_rate:.1f}%, Near={near_rate:.1f}%, Semantic={semantic_rate:.1f}%")
        
        dataset_stats.extend([
            {'Dataset': dataset, 'Metric': 'Exact Match (%)', 'Value': exact_rate},
            {'Dataset': dataset, 'Metric': 'Near Match (%)', 'Value': near_rate},
            {'Dataset': dataset, 'Metric': 'Semantic Match (%)', 'Value': semantic_rate}
        ])
    
    stats_df = pd.DataFrame(dataset_stats)
    
    # Create grouped bar plot
    ax = sns.barplot(data=stats_df, x='Dataset', y='Value', hue='Metric')
    
    plt.title('Performance Comparison Across Datasets', fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontweight='bold')
    plt.ylabel('Success Rate (%)', fontweight='bold')
    plt.legend(title='Match Type', title_fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'dataset_performance_comparison.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.close()

def plot_processing_time_vs_difficulty(df, save_path):
    """Analyze relationship between processing time and difficulty."""
    plt.figure(figsize=(12, 8))
    
    # Create violin plot
    sns.violinplot(data=df, x='Difficulty', y='Processing Time (s)', inner='box')
    
    plt.title('Processing Time Distribution by Question Difficulty', fontweight='bold', pad=20)
    plt.xlabel('Difficulty Level', fontweight='bold')
    plt.ylabel('Processing Time (seconds)', fontweight='bold')
    
    # Add median annotations
    for i, difficulty in enumerate(df['Difficulty'].unique()):
        subset = df[df['Difficulty'] == difficulty]
        median_val = subset['Processing Time (s)'].median()
        plt.text(i, median_val + 5, f'Median: {median_val:.1f}s', 
                ha='center', fontweight='bold', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path / 'processing_time_vs_difficulty.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.close()

def plot_success_rate_heatmap(df, save_path):
    """Create heatmap of success rates by difficulty and dataset."""
    # Prepare data for heatmap
    heatmap_data = []
    datasets = sorted(df['Dataset'].unique())
    difficulties = sorted(df['Difficulty'].unique())
    
    for dataset in datasets:
        for difficulty in difficulties:
            subset = df[(df['Dataset'] == dataset) & (df['Difficulty'] == difficulty)]
            if len(subset) > 0:
                semantic_rate = subset['Semantic Match'].mean() * 100
                heatmap_data.append({
                    'Dataset': dataset,
                    'Difficulty': difficulty,
                    'Semantic Match Rate': semantic_rate
                })
    
    if heatmap_data:  # Only create heatmap if we have data
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_df = heatmap_df.pivot(index='Difficulty', columns='Dataset', values='Semantic Match Rate')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Semantic Match Rate (%)'})
        
        plt.title('Semantic Match Success Rate by Dataset and Difficulty', fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontweight='bold')
        plt.ylabel('Difficulty Level', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path / 'success_rate_heatmap.png', dpi=600, bbox_inches='tight',
                    facecolor='white', edgecolor='none', format='png')
        plt.close()

def main():
    """Main function to generate all analysis plots."""
    # Set paths
    csv_path = Path('/mnt/data3/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/qa_dataset/reasoning_with_ragtool/Reasoning with tools answers.csv')
    save_path = csv_path.parent
    
    # Load data
    print("Loading and preparing data...")
    df = load_and_prepare_data(csv_path)
    
    # Generate plots
    print("Generating accuracy by level plot...")
    plot_match_types_by_difficulty(df, save_path)
    
    print("Generating processing time distribution plot...")
    plot_processing_time_distribution(df, save_path)
    
    print("Generating tool calls analysis plot...")
    plot_tool_calls_analysis(df, save_path)
    
    print("Generating dataset performance comparison plot...")
    plot_dataset_performance_comparison(df, save_path)
    
    print("Generating processing time vs difficulty plot...")
    plot_processing_time_vs_difficulty(df, save_path)
    
    print("Generating success rate heatmap...")
    plot_success_rate_heatmap(df, save_path)
    
    print("All plots generated successfully!")
    
    # Filter out outlier for consistent statistics with the plot
    df_filtered = df[df['Processing Time (s)'] <= 600]
    outlier_count = len(df) - len(df_filtered)
    
    # Print summary statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Total questions: {len(df)}")
    if outlier_count > 0:
        print(f"Questions after removing outliers (>600s): {len(df_filtered)} (removed {outlier_count} outlier{'s' if outlier_count > 1 else ''})")
    print(f"Datasets: {sorted(df['Dataset'].unique())}")
    print(f"Difficulty levels: {sorted(df['Difficulty'].unique())}")
    print(f"Overall semantic match rate: {df['Semantic Match'].mean()*100:.1f}%")
    print(f"Overall exact match rate: {df['Exact Match'].mean()*100:.1f}%")
    print(f"Overall near match rate: {df['Near Match'].mean()*100:.1f}%")
    print(f"Average processing time (excluding outliers): {df_filtered['Processing Time (s)'].mean():.1f}s")
    print(f"Median processing time (excluding outliers): {df_filtered['Processing Time (s)'].median():.1f}s")
    if outlier_count > 0:
        print(f"Average processing time (including all data): {df['Processing Time (s)'].mean():.1f}s")
        print(f"Median processing time (including all data): {df['Processing Time (s)'].median():.1f}s")
    
    # Print average processing time by difficulty (excluding outliers)
    print("\n=== AVERAGE PROCESSING TIME BY DIFFICULTY (excluding outliers) ===")
    for difficulty in sorted(df['Difficulty'].unique()):
        subset = df_filtered[df_filtered['Difficulty'] == difficulty]
        if len(subset) > 0:
            avg_time = subset['Processing Time (s)'].mean()
            print(f"{difficulty.capitalize()}: {avg_time:.1f} seconds")

if __name__ == "__main__":
    main()

# scripts/evaluate_enhanced.py

import os
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse

from inference import EnsembleInference

class EnhancedEvaluator:
    def __init__(self, config_path: str, checkpoint_dir: str):
        """Initialize evaluator with ensemble inference"""
        self.ensemble = EnsembleInference(config_path, checkpoint_dir)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def evaluate_on_dataset(self, test_df: pd.DataFrame, image_dir: str) -> Dict:
        """Evaluate ensemble on test dataset"""
        
        print(f"Evaluating on {len(test_df)} samples...")
        
        # Get predictions
        predictions_df = self.ensemble.predict_from_dataframe(test_df, image_dir)
        
        # Merge with ground truth
        merged_df = test_df.merge(predictions_df, left_on='Meme ID', right_on='meme_id', how='inner')
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(merged_df)
        
        # Generate visualizations
        self._create_evaluation_plots(merged_df, metrics)
        
        return metrics
    
    def _calculate_comprehensive_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {}
        
        # Single-class evaluation
        single_class_cols = {
            'sentiment': ('Overall Sentiment', 'sentiment_prediction'),
            'humor': ('Humor Mechanism', 'humor_prediction'),
            'sarcasm': ('Sarcasm Level', 'sarcasm_prediction'),
            'human_perception': ('Audience Perception', 'human_perception_prediction')
        }
        
        for task, (true_col, pred_col) in single_class_cols.items():
            if true_col in df.columns and pred_col in df.columns:
                # Filter out missing values
                mask = df[true_col].notna() & df[pred_col].notna()
                y_true = df.loc[mask, true_col]
                y_pred = df.loc[mask, pred_col]
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Per-class metrics
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average=None, zero_division=0
                )
                
                metrics[task] = {
                    'accuracy': accuracy,
                    'macro_f1': macro_f1,
                    'weighted_f1': weighted_f1,
                    'per_class_precision': precision,
                    'per_class_recall': recall,
                    'per_class_f1': f1,
                    'per_class_support': support,
                    'unique_labels': list(y_true.unique())
                }
        
        # Multi-label evaluation (simplified for this example)
        multi_label_cols = {
            'brands': ('Identified Brands', 'brands_predictions'),
            'context': ('Product Context', 'context_predictions'),
            'technical': ('Technical Concepts', 'technical_predictions')
        }
        
        for task, (true_col, pred_col) in multi_label_cols.items():
            if true_col in df.columns and pred_col in df.columns:
                # Simple overlap-based evaluation for multi-label
                overlaps = []
                for _, row in df.iterrows():
                    true_labels = set(str(row[true_col]).lower().split(','))
                    pred_labels = set(str(row[pred_col]).lower().split(','))
                    
                    # Clean labels
                    true_labels = {label.strip() for label in true_labels if label.strip() and label.strip() != 'nan'}
                    pred_labels = {label.strip() for label in pred_labels if label.strip() and label.strip() != 'nan'}
                    
                    if len(true_labels) > 0:
                        overlap = len(true_labels.intersection(pred_labels)) / len(true_labels.union(pred_labels))
                        overlaps.append(overlap)
                
                metrics[task] = {
                    'average_jaccard': np.mean(overlaps) if overlaps else 0.0,
                    'std_jaccard': np.std(overlaps) if overlaps else 0.0
                }
        
        # Overall score calculation
        key_metrics = []
        if 'human_perception' in metrics:
            key_metrics.append(metrics['human_perception']['accuracy'])
        if 'sentiment' in metrics:
            key_metrics.append(metrics['sentiment']['macro_f1'])
        if 'humor' in metrics:
            key_metrics.append(metrics['humor']['macro_f1'])
        if 'sarcasm' in metrics:
            key_metrics.append(metrics['sarcasm']['macro_f1'])
        
        metrics['overall_score'] = np.mean(key_metrics) if key_metrics else 0.0
        
        return metrics
    
    def _create_evaluation_plots(self, df: pd.DataFrame, metrics: Dict):
        """Create comprehensive evaluation visualizations"""
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall performance bar chart
        ax1 = axes[0, 0]
        task_scores = []
        task_names = []
        
        for task in ['human_perception', 'sentiment', 'humor', 'sarcasm']:
            if task in metrics:
                if 'accuracy' in metrics[task]:
                    task_scores.append(metrics[task]['accuracy'])
                    task_names.append(f"{task.replace('_', ' ').title()}\n(Accuracy)")
                elif 'macro_f1' in metrics[task]:
                    task_scores.append(metrics[task]['macro_f1'])
                    task_names.append(f"{task.replace('_', ' ').title()}\n(Macro F1)")
        
        bars = ax1.bar(task_names, task_scores, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax1.set_ylabel('Score')
        ax1.set_title('Performance by Task')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, task_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add target line at 89%
        ax1.axhline(y=0.89, color='red', linestyle='--', alpha=0.7, label='Target (89%)')
        ax1.legend()
        
        # 2. Confusion matrix for human_perception (most important)
        ax2 = axes[0, 1]
        if 'human_perception' in metrics and 'human_perception_prediction' in df.columns:
            true_col = 'Audience Perception'
            pred_col = 'human_perception_prediction'
            
            mask = df[true_col].notna() & df[pred_col].notna()
            y_true = df.loc[mask, true_col]
            y_pred = df.loc[mask, pred_col]
            
            labels = sorted(list(set(y_true) | set(y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=ax2)
            ax2.set_title('Human Perception\nConfusion Matrix (Normalized)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
        
        # 3. Multi-label performance
        ax3 = axes[1, 0]
        multilabel_tasks = ['brands', 'context', 'technical']
        multilabel_scores = []
        
        for task in multilabel_tasks:
            if task in metrics:
                multilabel_scores.append(metrics[task]['average_jaccard'])
            else:
                multilabel_scores.append(0.0)
        
        bars3 = ax3.bar([t.title() for t in multilabel_tasks], multilabel_scores, 
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax3.set_ylabel('Average Jaccard Score')
        ax3.set_title('Multi-label Task Performance')
        ax3.set_ylim(0, 1)
        
        for bar, score in zip(bars3, multilabel_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance distribution
        ax4 = axes[1, 1]
        all_scores = []
        all_labels = []
        
        for task in ['human_perception', 'sentiment', 'humor', 'sarcasm']:
            if task in metrics:
                if 'accuracy' in metrics[task]:
                    all_scores.append(metrics[task]['accuracy'])
                    all_labels.append(task.replace('_', ' ').title())
        
        # Add multi-label scores
        for task in multilabel_tasks:
            if task in metrics:
                all_scores.append(metrics[task]['average_jaccard'])
                all_labels.append(f"{task.title()} (Jaccard)")
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_scores)))
        wedges, texts, autotexts = ax4.pie(all_scores, labels=all_labels, autopct='%1.1f%%', colors=colors)
        ax4.set_title('Score Distribution Across Tasks')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.config.get('output_dir', 'results'), 'evaluation_results.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Evaluation plots saved to: {plot_path}")
        
        plt.show()
    
    def print_detailed_report(self, metrics: Dict):
        """Print detailed evaluation report"""
        
        print("\n" + "="*80)
        print("DETAILED EVALUATION REPORT")
        print("="*80)
        
        # Overall performance
        overall_score = metrics.get('overall_score', 0.0) * 100
        print(f"\nOVERALL PERFORMANCE: {overall_score:.2f}%")
        
        if overall_score >= 89.0:
            print("SUCCESS! Target of 89%+ accuracy achieved!")
        else:
            print(f"Current performance: {overall_score:.2f}% (Target: 89%+)")
        
        print(f"\n{'Task':<20} {'Metric':<15} {'Score':<10} {'Status':<10}")
        print("-" * 60)
        
        # Single-class tasks
        for task in ['human_perception', 'sentiment', 'humor', 'sarcasm']:
            if task in metrics:
                task_metrics = metrics[task]
                if 'accuracy' in task_metrics:
                    score = task_metrics['accuracy'] * 100
                    metric_name = 'Accuracy'
                else:
                    score = task_metrics['macro_f1'] * 100
                    metric_name = 'Macro F1'
                
                status = "✅ Good" if score >= 85 else "⚠️ Needs Work"
                print(f"{task.replace('_', ' ').title():<20} {metric_name:<15} {score:<10.2f}% {status}")
        
        # Multi-label tasks
        print(f"\n{'Multi-label Task':<20} {'Jaccard Score':<15} {'Performance':<10}")
        print("-" * 50)
        
        for task in ['brands', 'context', 'technical']:
            if task in metrics:
                score = metrics[task]['average_jaccard'] * 100
                status = "✅ Good" if score >= 70 else "⚠️ Needs Work"
                print(f"{task.title():<20} {score:<15.2f}% {status}")
        
        # Detailed per-class performance
        print(f"\nPER-CLASS PERFORMANCE:")
        print("-" * 40)
        
        for task in ['human_perception', 'sentiment', 'humor', 'sarcasm']:
            if task in metrics and 'per_class_f1' in metrics[task]:
                print(f"\n{task.replace('_', ' ').title()}:")
                
                task_metrics = metrics[task]
                labels = task_metrics['unique_labels']
                f1_scores = task_metrics['per_class_f1']
                
                for label, f1 in zip(labels, f1_scores):
                    print(f"  {label:<25}: {f1*100:>6.2f}%")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        print("-" * 20)
        
        if overall_score < 89:
            print("• Consider increasing training epochs or learning rate tuning")
            print("• Try different loss weights for underperforming tasks")
            print("• Add more data augmentation techniques")
            print("• Experiment with larger model architectures")
        
        # Check specific weak areas
        weak_tasks = []
        for task in ['human_perception', 'sentiment', 'humor', 'sarcasm']:
            if task in metrics:
                score = metrics[task].get('accuracy', metrics[task].get('macro_f1', 0))
                if score < 0.85:
                    weak_tasks.append(task)
        
        if weak_tasks:
            print(f"• Focus on improving: {', '.join(weak_tasks)}")
            print("• Consider task-specific data augmentation")
            print("• Increase loss weights for these tasks")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Model Evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Model checkpoint directory')
    parser.add_argument('--test_csv', type=str, required=True, help='Test dataset CSV file')
    parser.add_argument('--image_dir', type=str, required=True, help='Test images directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    test_df = pd.read_excel(args.test_csv)  # Assuming Excel format like training
    print(f"Loaded {len(test_df)} test samples")
    
    # Initialize evaluator
    evaluator = EnhancedEvaluator(args.config, args.checkpoint_dir)
    
    # Run evaluation
    print("Starting comprehensive evaluation...")
    metrics = evaluator.evaluate_on_dataset(test_df, args.image_dir)
    
    # Print detailed report
    evaluator.print_detailed_report(metrics)
    
    # Save metrics to file
    import json
    metrics_file = os.path.join(args.output_dir, 'evaluation_metrics.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_metrics = {}
    for task, task_metrics in metrics.items():
        if isinstance(task_metrics, dict):
            json_task_metrics = {}
            for key, value in task_metrics.items():
                if isinstance(value, np.ndarray):
                    json_task_metrics[key] = value.tolist()
                else:
                    json_task_metrics[key] = value
            json_metrics[task] = json_task_metrics
        else:
            json_metrics[task] = task_metrics
    
    with open(metrics_file, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    
    print(f"\nDetailed metrics saved to: {metrics_file}")

if __name__ == '__main__':
    main()
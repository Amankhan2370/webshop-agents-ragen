"""
Metrics tracking and visualization for WebShop/WebArena evaluation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class MetricsTracker:
    """
    Track and visualize training/evaluation metrics
    """
    
    def __init__(self, experiment_dir: str):
        """Initialize metrics tracker"""
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.episode_metrics = []
        self.evaluation_metrics = []
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'steps': [],
            'values': []
        }
        
        # Create metrics file
        self.metrics_file = self.experiment_dir / "metrics.json"
        
    def add_episode(self, metrics: Dict[str, Any]):
        """Add metrics from single training episode"""
        self.episode_metrics.append({
            'episode': len(self.episode_metrics) + 1,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
        
        # Update training stats
        self.training_stats['episodes'].append(len(self.episode_metrics))
        self.training_stats['rewards'].append(metrics.get('reward', 0))
        self.training_stats['success_rates'].append(metrics.get('success', 0))
        self.training_stats['steps'].append(metrics.get('steps', 0))
        self.training_stats['values'].append(metrics.get('v_star_final', 0))
    
    def add_evaluation(self, episode: int, eval_results: Dict[str, Any]):
        """Add evaluation results"""
        self.evaluation_metrics.append({
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **eval_results
        })
    
    def calculate_statistics(self, window: int = 100) -> Dict[str, float]:
        """Calculate moving statistics"""
        if len(self.training_stats['rewards']) < window:
            window = len(self.training_stats['rewards'])
        
        if window == 0:
            return {}
        
        recent_rewards = self.training_stats['rewards'][-window:]
        recent_success = self.training_stats['success_rates'][-window:]
        recent_steps = self.training_stats['steps'][-window:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_success_rate': np.mean(recent_success) * 100,
            'mean_steps': np.mean(recent_steps),
            'improvement_rate': self._calculate_improvement_rate(recent_rewards)
        }
    
    def _calculate_improvement_rate(self, values: List[float]) -> float:
        """Calculate improvement rate over time"""
        if len(values) < 2:
            return 0.0
        
        # Fit linear regression
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        return coefficients[0]  # Slope
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves"""
        if len(self.episode_metrics) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Rewards over time
        ax1 = axes[0, 0]
        episodes = self.training_stats['episodes']
        rewards = self.training_stats['rewards']
        
        # Plot raw rewards
        ax1.plot(episodes, rewards, alpha=0.3, label='Raw')
        
        # Plot smoothed rewards
        if len(rewards) > 10:
            window = min(50, len(rewards) // 5)
            smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean()
            ax1.plot(episodes, smoothed, label=f'Smoothed (w={window})', linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Success rate over time
        ax2 = axes[0, 1]
        success_rates = [s * 100 for s in self.training_stats['success_rates']]
        
        # Calculate rolling success rate
        window = min(50, len(success_rates) // 5) if len(success_rates) > 10 else 1
        rolling_success = pd.Series(success_rates).rolling(window, min_periods=1).mean()
        
        ax2.plot(episodes, rolling_success, color='green', linewidth=2)
        ax2.fill_between(episodes, rolling_success, alpha=0.3, color='green')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title(f'Success Rate (Rolling Window={window})')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3)
        
        # 3. Episode length over time
        ax3 = axes[1, 0]
        steps = self.training_stats['steps']
        
        ax3.plot(episodes, steps, alpha=0.5, color='orange')
        if len(steps) > 10:
            smoothed_steps = pd.Series(steps).rolling(20, min_periods=1).mean()
            ax3.plot(episodes, smoothed_steps, color='red', linewidth=2, label='Smoothed')
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.set_title('Episode Length')
        ax3.grid(True, alpha=0.3)
        
        # 4. Value estimates over time
        ax4 = axes[1, 1]
        values = self.training_stats['values']
        
        if any(v != 0 for v in values):  # Only plot if we have V* values
            ax4.plot(episodes, values, alpha=0.5, color='purple')
            if len(values) > 10:
                smoothed_values = pd.Series(values).rolling(20, min_periods=1).mean()
                ax4.plot(episodes, smoothed_values, color='darkviolet', linewidth=2)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('V* Estimate')
            ax4.set_title('Value Function Estimates')
        else:
            ax4.text(0.5, 0.5, 'No V* data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Training Progress', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.experiment_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to {save_path}")
    
    def plot_evaluation_results(self, save_path: Optional[str] = None):
        """Plot evaluation results"""
        if len(self.evaluation_metrics) == 0:
            print("No evaluation data to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract data
        eval_episodes = [e['episode'] for e in self.evaluation_metrics]
        success_rates = [e.get('success_rate', 0) for e in self.evaluation_metrics]
        avg_rewards = [e.get('avg_reward', 0) for e in self.evaluation_metrics]
        
        # 1. Success rate evolution
        ax1 = axes[0]
        ax1.plot(eval_episodes, success_rates, marker='o', linewidth=2, 
                markersize=8, color='green')
        ax1.set_xlabel('Training Episode')
        ax1.set_ylabel('Evaluation Success Rate (%)')
        ax1.set_title('Evaluation Performance Over Training')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # Add target line if specified
        target = 75  # Default target
        ax1.axhline(y=target, color='red', linestyle='--', alpha=0.5, 
                   label=f'Target ({target}%)')
        ax1.legend()
        
        # 2. Reward evolution
        ax2 = axes[1]
        ax2.plot(eval_episodes, avg_rewards, marker='s', linewidth=2,
                markersize=8, color='blue')
        ax2.set_xlabel('Training Episode')
        ax2.set_ylabel('Average Evaluation Reward')
        ax2.set_title('Reward Evolution During Training')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Evaluation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.experiment_dir / "evaluation_results.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation results saved to {save_path}")
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create summary statistics table"""
        stats = self.calculate_statistics()
        
        if self.evaluation_metrics:
            final_eval = self.evaluation_metrics[-1]
            eval_success = final_eval.get('success_rate', 0)
            eval_reward = final_eval.get('avg_reward', 0)
        else:
            eval_success = 0
            eval_reward = 0
        
        summary_data = {
            'Metric': [
                'Total Episodes',
                'Mean Reward (last 100)',
                'Mean Success Rate (last 100)',
                'Mean Steps (last 100)',
                'Final Eval Success Rate',
                'Final Eval Reward',
                'Improvement Rate'
            ],
            'Value': [
                len(self.episode_metrics),
                f"{stats.get('mean_reward', 0):.3f} ± {stats.get('std_reward', 0):.3f}",
                f"{stats.get('mean_success_rate', 0):.1f}%",
                f"{stats.get('mean_steps', 0):.1f}",
                f"{eval_success:.1f}%",
                f"{eval_reward:.3f}",
                f"{stats.get('improvement_rate', 0):.6f}"
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def save(self):
        """Save all metrics to file"""
        data = {
            'episode_metrics': self.episode_metrics,
            'evaluation_metrics': self.evaluation_metrics,
            'training_stats': self.training_stats,
            'summary': self.calculate_statistics()
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Also save summary table
        summary_df = self.create_summary_table()
        summary_df.to_csv(self.experiment_dir / "summary.csv", index=False)
        
        print(f"Metrics saved to {self.metrics_file}")
    
    def load(self):
        """Load metrics from file"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
                
            self.episode_metrics = data.get('episode_metrics', [])
            self.evaluation_metrics = data.get('evaluation_metrics', [])
            self.training_stats = data.get('training_stats', {
                'episodes': [],
                'rewards': [],
                'success_rates': [],
                'steps': [],
                'values': []
            })
            
            print(f"Metrics loaded from {self.metrics_file}")
        else:
            print(f"No metrics file found at {self.metrics_file}")


def calculate_efficiency(steps: int, optimal_steps: int = 5) -> float:
    """Calculate efficiency score based on steps taken"""
    if steps <= optimal_steps:
        return 1.0
    else:
        return optimal_steps / steps


def calculate_subtask_completion_rate(
    task_progress: Dict[str, bool]
) -> float:
    """Calculate percentage of completed subtasks"""
    if not task_progress:
        return 0.0
    
    completed = sum(1 for v in task_progress.values() if v)
    total = len(task_progress)
    
    return (completed / total) * 100


def compare_agents(
    results_dict: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """Compare performance of multiple agents"""
    comparison_data = []
    
    for agent_name, results in results_dict.items():
        comparison_data.append({
            'Agent': agent_name,
            'Success Rate': results.get('success_rate', 0),
            'Avg Reward': results.get('avg_reward', 0),
            'Avg Steps': results.get('avg_steps', 0),
            'Efficiency': results.get('efficiency', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Success Rate', ascending=False)
    
    return df
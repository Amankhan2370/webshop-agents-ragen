#!/usr/bin/env python3
"""
Training script for WebShop RAGEN agent
Run this to train the agent on WebShop tasks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
import time

from agents.webshop_agent import WebShopRAGEN
from environments.webshop_env import WebShopEnvironment
from evaluation.metrics import MetricsTracker
from utils.logging import setup_logger, log_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_dir(base_dir: str = "experiments/results") -> str:
    """Create directory for experiment results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"webshop_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return str(exp_dir)


def train_webshop(config: dict, args):
    """Main training function"""
    
    # Setup
    logger = setup_logger("WebShop_Training")
    logger.info("Starting WebShop RAGEN Training")
    logger.info(f"Config: {json.dumps(config['webshop'], indent=2)}")
    
    # Set seed
    seed = config['webshop'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Create experiment directory
    exp_dir = create_experiment_dir()
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Save config
    with open(Path(exp_dir) / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Initialize environment
    env_config = config['webshop'].get('environment', {})
    env = WebShopEnvironment(env_config)
    logger.info(f"WebShop environment initialized with {len(env.products)} products")
    
    # Initialize agent
    agent_config = config['webshop']
    agent = WebShopRAGEN(agent_config)
    logger.info(f"RAGEN agent initialized on {agent.device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in agent.model.parameters()):,}")
    
    # Initialize metrics tracker
    metrics = MetricsTracker(exp_dir)
    
    # Training settings
    num_episodes = config['webshop']['training'].get('num_episodes', 1000)
    eval_interval = config['webshop']['training'].get('eval_interval', 100)
    save_interval = config['webshop']['training'].get('save_interval', 200)
    early_stopping_patience = config['webshop']['training'].get('early_stopping_patience', 500)
    target_success_rate = config['webshop']['training'].get('target_success_rate', 0.75)
    
    # Training loop
    logger.info(f"Starting training for {num_episodes} episodes")
    logger.info("=" * 60)
    
    best_success_rate = 0.0
    episodes_without_improvement = 0
    training_start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        # Train one episode
        episode_start_time = time.time()
        stats = agent.train_episode(env)
        episode_time = time.time() - episode_start_time
        
        # Track metrics
        metrics.add_episode(stats)
        
        # Log progress
        if episode % 10 == 0:
            recent_rewards = agent.episode_rewards[-10:]
            recent_success = agent.success_rate[-10:]
            
            logger.info(
                f"Episode {episode:4d} | "
                f"Reward: {np.mean(recent_rewards):6.2f} | "
                f"Success: {np.mean(recent_success)*100:5.1f}% | "
                f"Steps: {stats['steps']:3d} | "
                f"Time: {episode_time:.2f}s"
            )
        
        # Evaluation
        if episode % eval_interval == 0:
            logger.info("-" * 40)
            logger.info("Running evaluation...")
            
            eval_results = agent.evaluate(env, num_episodes=50)
            metrics.add_evaluation(episode, eval_results)
            
            # Check for improvement
            if eval_results['success_rate'] > best_success_rate:
                best_success_rate = eval_results['success_rate']
                episodes_without_improvement = 0
                
                # Save best model
                best_model_path = Path(exp_dir) / "best_model.pt"
                agent.save(str(best_model_path))
                logger.info(f"New best model saved! Success rate: {best_success_rate:.1f}%")
            else:
                episodes_without_improvement += eval_interval
            
            # Check early stopping
            if episodes_without_improvement >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {episode} episodes")
                break
            
            # Check target reached
            if eval_results['success_rate'] >= target_success_rate * 100:
                logger.info(f"🎉 Target success rate reached: {eval_results['success_rate']:.1f}%")
                break
            
            logger.info("-" * 40)
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = Path(exp_dir) / f"checkpoint_ep{episode}.pt"
            agent.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved at episode {episode}")
        
        # Save metrics periodically
        if episode % 100 == 0:
            metrics.save()
    
    # Training complete
    training_time = time.time() - training_start_time
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Total episodes: {episode}")
    logger.info(f"Training time: {training_time/3600:.2f} hours")
    logger.info(f"Best success rate: {best_success_rate:.1f}%")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_results = agent.evaluate(env, num_episodes=100)
    metrics.add_evaluation(episode, final_results)
    
    # Save final model
    final_model_path = Path(exp_dir) / "final_model.pt"
    agent.save(str(final_model_path))
    
    # Save final metrics
    metrics.save()
    
    # Generate summary
    summary = {
        'experiment_dir': exp_dir,
        'total_episodes': episode,
        'training_time_hours': training_time / 3600,
        'best_success_rate': best_success_rate,
        'final_success_rate': final_results['success_rate'],
        'final_avg_reward': final_results['avg_reward'],
        'final_avg_steps': final_results['avg_steps'],
        'model_parameters': sum(p.numel() for p in agent.model.parameters()),
        'config': config['webshop']
    }
    
    with open(Path(exp_dir) / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {exp_dir}")
    
    # Generate plots
    try:
        metrics.plot_training_curves()
        logger.info("Training curves plotted")
    except Exception as e:
        logger.warning(f"Failed to plot training curves: {e}")
    
    return summary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train WebShop RAGEN Agent")
    parser.add_argument(
        '--config',
        type=str,
        default='training/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override device if specified
    if args.device:
        config['webshop']['device'] = args.device
    
    # Debug mode
    if args.debug:
        config['webshop']['training']['num_episodes'] = 10
        config['webshop']['training']['eval_interval'] = 5
    
    # Train
    summary = train_webshop(config, args)
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Success Rate: {summary['final_success_rate']:.1f}%")
    print(f"Best Success Rate: {summary['best_success_rate']:.1f}%")
    print(f"Training Time: {summary['training_time_hours']:.2f} hours")
    print(f"Results Directory: {summary['experiment_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Main entry point for WebShop-WebArena RAGEN project
Run this file to train and evaluate the agents
"""

import argparse
import sys
from pathlib import Path


def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║         WebShop-WebArena RAGEN Implementation           ║
    ║     Reasoning via A*-guided Planning with A*PO          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point with menu"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="WebShop-WebArena RAGEN: Web Navigation with A*PO",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'compare', 'demo', 'all'],
        help="""
        Command to run:
        - train: Train RAGEN on WebShop
        - evaluate: Evaluate on WebArena
        - compare: Compare with baselines
        - demo: Run interactive demo
        - all: Run all experiments
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='training/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (for evaluation)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/results',
        help='Directory to save results'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with fewer episodes'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        print("\n🚀 Starting WebShop Training...")
        from training.train_webshop import main as train_main
        sys.argv = [
            'train_webshop.py',
            '--config', args.config,
            '--output_dir', args.output_dir
        ]
        if args.debug:
            sys.argv.append('--debug')
        train_main()
        
    elif args.command == 'evaluate':
        print("\n📊 Starting WebArena Evaluation...")
        from evaluation.evaluate_webarena import main as eval_main
        
        # Find checkpoint if not provided
        if args.checkpoint is None:
            checkpoint_path = Path(args.output_dir) / 'webshop' / 'best_model.pt'
            if not checkpoint_path.exists():
                checkpoint_path = Path(args.output_dir) / 'webshop' / 'final_model.pt'
            args.checkpoint = str(checkpoint_path)
        
        sys.argv = [
            'evaluate_webarena.py',
            '--config', args.config,
            '--checkpoint', args.checkpoint,
            '--output_dir', args.output_dir
        ]
        eval_main()
        
    elif args.command == 'compare':
        print("\n⚖️ Comparing with Baselines...")
        from agents.baseline_agents import evaluate_baseline_agents
        from environments.webshop_env import WebShopEnvironment
        
        env = WebShopEnvironment()
        results = evaluate_baseline_agents(env, num_episodes=100)
        
        print("\n" + "="*60)
        print("Baseline Comparison Results")
        print("="*60)
        for agent_type, metrics in results.items():
            print(f"\n{agent_type.upper()}:")
            print(f"  Success Rate: {metrics['success_rate']:.1f}%")
            print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
            print(f"  Avg Steps: {metrics['avg_steps']:.1f}")
        
    elif args.command == 'demo':
        print("\n🎮 Starting Interactive Demo...")
        run_interactive_demo(args)
        
    elif args.command == 'all':
        print("\n🔬 Running All Experiments...")
        from experiments.run_all import main as run_all_main
        sys.argv = [
            'run_all.py',
            '--config', args.config,
            '--output_dir', args.output_dir
        ]
        run_all_main()
    
    print("\n✅ Command completed successfully!")


def run_interactive_demo(args):
    """Run interactive demo of the agent"""
    from agents.webshop_agent import WebShopRAGEN
    from environments.webshop_env import WebShopEnvironment
    import torch
    
    print("\n" + "="*60)
    print("Interactive WebShop Demo")
    print("="*60)
    
    # Load agent
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        agent = WebShopRAGEN({'model': {'hidden_dim': 256}})
        agent.load(args.checkpoint)
    else:
        print("No checkpoint found, using untrained agent")
        agent = WebShopRAGEN({'model': {'hidden_dim': 256}})
    
    # Create environment
    env = WebShopEnvironment()
    
    # Interactive loop
    while True:
        print("\nSelect a shopping goal:")
        print("1. Buy wireless headphones under $100")
        print("2. Find a coffee maker under $50")
        print("3. Purchase running shoes under $80")
        print("4. Custom goal")
        print("5. Quit")
        
        choice = input("\nYour choice (1-5): ")
        
        if choice == '5':
            break
        
        if choice == '4':
            goal = input("Enter custom goal: ")
            state = env.reset()
            # Update goal manually
            env.current_goal = {'description': goal, 'budget': 100}
        else:
            goal_index = int(choice) - 1 if choice in '123' else 0
            state = env.reset(goal_index)
        
        print(f"\nGoal: {state.goal}")
        print("Starting shopping session...\n")
        
        total_reward = 0
        for step in range(env.max_steps):
            # Display current state
            print(f"Step {step+1}: {state.to_text()}")
            
            # Get agent action
            action, action_arg = agent.select_action(state, epsilon=0.0)
            
            # Display action
            action_name = list(env.ACTIONS.keys())[list(env.ACTIONS.values()).index(action)]
            print(f"Agent action: {action_name} {action_arg or ''}")
            
            # Execute action
            state, reward, done, info = env.step(action, action_arg)
            total_reward += reward
            
            if done:
                if info.get('success'):
                    print("\n🎉 SUCCESS! Task completed!")
                else:
                    print("\n❌ Task failed or timed out")
                print(f"Total reward: {total_reward:.2f}")
                break
            
            # Optional: pause between steps
            input("\nPress Enter to continue...")
    
    print("\nThank you for using the demo!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
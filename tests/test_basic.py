"""
Basic tests for WebShop-WebArena RAGEN implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from pathlib import Path

# Import your modules
from environments.webshop_env import WebShopEnvironment
from environments.webarena_env import WebArenaEnvironment

# Import the correct class names
from agents.webshop_agent import WebShopRAGEN as WebShopRAGENAgent
from agents.baseline_agents import RandomAgent, RuleBasedAgent
from algorithms.astarpo import AStarPO, AStarPOConfig

# Import TransformerLM instead of SimpleTransformer
from models.transformer import TransformerLM as SimpleTransformer

from evaluation.metrics import MetricsTracker
from evaluation.failure_analysis import FailureAnalyzer

class TestEnvironments:
    """Test environment implementations"""
    
    def test_webshop_init(self):
        """Test WebShop environment initialization"""
        # WebShopEnvironment doesn't take max_steps in __init__
        env = WebShopEnvironment()
        assert env.max_steps == 10  # Default value
    
    def test_webshop_reset(self):
        """Test WebShop environment reset"""
        env = WebShopEnvironment()
        # reset() returns only state, not (state, info)
        state = env.reset()
        assert state is not None
        # Check that state has been reset properly
        assert state.step_count == 0
    
    def test_webshop_step(self):
        """Test WebShop environment step"""
        env = WebShopEnvironment()
        state = env.reset()
        
        # Take a random action - step returns 4 values, not 5
        next_state, reward, done, info = env.step(0)
        assert next_state is not None
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)

    def test_webarena_init(self):
        """Test WebArena environment initialization"""
        # WebArenaEnvironment doesn't take max_steps in __init__
        env = WebArenaEnvironment()
        assert env.max_steps == 30  # Default value

class TestAgents:
    """Test agent implementations"""
    
    def test_random_agent(self):
        """Test random agent"""
        # RandomAgent in baseline_agents.py expects action_space dict
        action_space = {'search': 0, 'click': 1, 'buy': 2}
        agent = RandomAgent(action_space)
        action = agent.act(None)
        assert action in [0, 1, 2]
    
    def test_rule_based_agent(self):
        """Test rule-based agent"""
        agent = RuleBasedAgent()
        
        # Create proper MockState class that works with RuleBasedAgent
        class MockState:
            def to_text(self):
                return '[Search Page] Goal: Buy headphones'
        
        state = MockState()
        action = agent.act(state)
        # RuleBasedAgent returns 0 for search action on search page
        assert action == 0  # Should be search action
    
    def test_webshop_ragen_init(self):
        """Test WebShop RAGEN agent initialization"""
        # WebShopRAGEN requires a config dict
        config = {
            'model': {
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'vocab_size': 50000
            },
            'beta': 0.3,
            'gamma': 0.95,
            'num_samples': 10,
            'learning_rate': 1e-4
        }
        agent = WebShopRAGENAgent(config)
        assert agent.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestAlgorithms:
    """Test algorithm implementations"""
    
    def test_astarpo_init(self):
        """Test A*PO initialization"""
        # AStarPO takes config, not model
        config = AStarPOConfig(
            beta=0.3,
            gamma=0.95,
            learning_rate=1e-4
        )
        algo = AStarPO(config)
        assert algo.config.learning_rate == 1e-4
    
    def test_compute_returns(self):
        """Test return computation"""
        config = AStarPOConfig(
            gamma=0.99
        )
        algo = AStarPO(config)
        
        # Create tensors for compute_returns
        rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        dones = torch.tensor([[0, 0, 0, 1]])
        
        returns = algo.compute_returns(rewards, dones)
        assert returns.shape == rewards.shape
        assert returns[0, 0] > returns[0, -1]  # First return should be largest

class TestModels:
    """Test model implementations"""
    
    def test_transformer_init(self):
        """Test transformer initialization"""
        model = SimpleTransformer(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4
        )
        assert model.hidden_size == 256
        assert model.num_layers == 4
    
    def test_model_device(self):
        """Test model device placement"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SimpleTransformer(
            vocab_size=100,
            hidden_size=128
        )
        model = model.to(device)
        
        # Check if model is on correct device
        dummy_input = torch.randint(0, 100, (1, 10)).to(device)
        output = model(dummy_input)
        assert output['logits'].device.type == device

class TestIntegration:
    """Test integration between components"""
    
    def test_training_step(self):
        """Test a single training step"""
        env = WebShopEnvironment()
        
        # WebShopRAGEN requires config
        config = {
            'model': {
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'vocab_size': 50000
            },
            'beta': 0.3,
            'gamma': 0.95,
            'num_samples': 10,
            'learning_rate': 1e-4
        }
        agent = WebShopRAGENAgent(config)
        
        state = env.reset()  # No unpacking
        action, action_arg = agent.select_action(state)
        next_state, reward, done, info = env.step(action, action_arg)
        
        # Check that action selection works
        assert isinstance(action, (int, np.integer))
        assert action >= 0
    
    def test_evaluation(self):
        """Test evaluation pipeline"""
        env = WebShopEnvironment()
        # RandomAgent needs action_space dict
        action_space = {'search': 0, 'click': 1, 'buy': 2, 'back': 3, 'next': 4, 'prev': 5}
        agent = RandomAgent(action_space)
        
        total_reward = 0
        state = env.reset()  # No unpacking
        
        for _ in range(10):
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        
        assert isinstance(total_reward, (int, float))

class TestUtils:
    """Test utility functions"""
    
    def test_metrics_tracker(self):
        """Test metrics tracking"""
        # MetricsTracker requires experiment_dir
        tracker = MetricsTracker('test_experiment')
        
        # Add some metrics
        tracker.add_episode({
            'reward': 10.5,
            'success': True,
            'steps': 5
        })
        tracker.add_episode({
            'reward': 15.2,
            'success': False,
            'steps': 8
        })
        
        stats = tracker.calculate_statistics()
        assert 'mean_reward' in stats
        assert stats['mean_reward'] == pytest.approx(12.85, 0.01)
    
    def test_failure_analyzer(self):
        """Test failure analysis"""
        analyzer = FailureAnalyzer()
        
        # Create test trajectory
        trajectory = [
            {'state': 'search', 'action': 0, 'reward': 0},
            {'state': 'results', 'action': 1, 'reward': 0}
        ]
        
        # Create dummy task
        class DummyTask:
            def __init__(self):
                self.id = 'test_task'
                self.complexity = 'easy'
        
        task = DummyTask()
        
        # analyze_trajectory returns failure info
        failure_info = analyzer.analyze_trajectory(trajectory, None, task)
        
        assert 'type' in failure_info
        assert failure_info is not None

def test_checkpoint_saving():
    """Test checkpoint saving and loading"""
    config = {
        'model': {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8,
            'vocab_size': 50000
        },
        'beta': 0.3,
        'gamma': 0.95,
        'num_samples': 10,
        'learning_rate': 1e-4
    }
    agent = WebShopRAGENAgent(config)
    
    # Create checkpoints directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
    agent.save(str(checkpoint_path))
    
    assert checkpoint_path.exists()
    
    # Load checkpoint
    new_agent = WebShopRAGENAgent(config)
    new_agent.load(str(checkpoint_path))
    
    # Clean up
    checkpoint_path.unlink()

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
"""
A* Policy Optimization (A*PO) Algorithm
Replaces PPO/GRPO with optimal value estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class AStarPOConfig:
    """Configuration for A*PO algorithm"""
    beta: float = 0.3  # KL penalty coefficient
    gamma: float = 0.95  # Discount factor
    num_samples: int = 10  # Samples for V* estimation
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    value_clip: float = 10.0
    advantage_normalization: bool = True
    use_gae: bool = True  # Generalized Advantage Estimation
    gae_lambda: float = 0.95


class AStarPO:
    """
    A* Policy Optimization
    
    Key innovation: Uses V*(s) = β log E[exp(r(s,a)/β)] for optimal value estimation
    instead of learned value function as in PPO
    """
    
    def __init__(self, config: AStarPOConfig):
        self.config = config
        self.beta = config.beta
        self.gamma = config.gamma
        self.num_samples = config.num_samples
        
    def compute_v_star(
        self,
        states: torch.Tensor,
        policy_model: nn.Module,
        environment,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute V*(s) = β log E[exp(r(s,a)/β)]
        
        This is the key innovation of A*PO: computing optimal value
        through sampling rather than learning it
        """
        batch_size = states.shape[0]
        v_star_values = torch.zeros(batch_size, device=device)
        
        with torch.no_grad():
            for i in range(batch_size):
                state = states[i:i+1]
                values = []
                
                # Sample multiple trajectories from current state
                for _ in range(self.config.num_samples):
                    # Get action distribution from policy
                    action_logits = policy_model(state)['action_logits']
                    action_probs = F.softmax(action_logits, dim=-1)
                    
                    # Sample action
                    action = torch.multinomial(action_probs, 1)
                    
                    # Simulate one-step lookahead (simplified)
                    # In practice, this would use environment.step()
                    expected_reward = torch.randn(1).item() * 0.5  # Placeholder
                    
                    # Estimate future value (recursive in full implementation)
                    future_value = self.gamma * torch.randn(1).item() * 0.3
                    
                    total_value = expected_reward + future_value
                    values.append(total_value)
                
                # Compute V* using log-sum-exp for numerical stability
                values_tensor = torch.tensor(values, device=device)
                v_star = self.beta * torch.logsumexp(values_tensor / self.beta, dim=0)
                v_star_values[i] = v_star
        
        return v_star_values
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        v_star: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute A*(s,a) = r(s,a) - V*(s)
        
        If V* not provided, uses GAE for advantage estimation
        """
        device = rewards.device
        batch_size, seq_len = rewards.shape
        
        if v_star is not None:
            # Use A*PO advantage: A*(s,a) = r(s,a) - V*(s)
            advantages = rewards - v_star.unsqueeze(1)
            returns = rewards  # Simple returns for A*PO
        
        elif self.config.use_gae:
            # Use Generalized Advantage Estimation as fallback
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            
            gae = 0
            next_value = 0
            
            for t in reversed(range(seq_len)):
                if t == seq_len - 1:
                    next_value = 0
                else:
                    next_value = values[:, t + 1]
                
                delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
                gae = delta + self.gamma * self.config.gae_lambda * (1 - dones[:, t]) * gae
                advantages[:, t] = gae
                returns[:, t] = advantages[:, t] + values[:, t]
        
        else:
            # Simple advantage calculation
            returns = self.compute_returns(rewards, dones)
            advantages = returns - values
        
        # Normalize advantages
        if self.config.advantage_normalization:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute discounted returns"""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        
        for i in range(batch_size):
            G = 0
            for t in reversed(range(seq_len)):
                if dones[i, t]:
                    G = 0
                G = rewards[i, t] + self.gamma * G
                returns[i, t] = G
        
        return returns
    
    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
        policy_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute policy loss using A*PO objective
        
        Unlike PPO which uses clipped surrogate objective,
        A*PO uses MSE loss with advantages from V*
        """
        # Get current policy log probabilities
        outputs = policy_model(states)
        action_logits = outputs['action_logits']
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Gather log probs for taken actions
        action_log_probs = torch.log(action_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        
        # A*PO uses regression-style update instead of PPO's ratio
        # This is more stable when advantages come from V*
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Add KL penalty to prevent policy from deviating too much
        kl_div = (old_log_probs - action_log_probs).mean()
        total_loss = policy_loss + self.beta * kl_div
        
        return total_loss
    
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute value function loss"""
        # Clip value predictions
        values_clipped = values.clamp(-self.config.value_clip, self.config.value_clip)
        
        # MSE loss
        value_loss = F.mse_loss(values_clipped, returns.detach())
        
        return value_loss
    
    def update(
        self,
        trajectories: List[Dict],
        policy_model: nn.Module,
        value_model: Optional[nn.Module],
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Update policy using A*PO algorithm
        
        Args:
            trajectories: List of trajectory dictionaries
            policy_model: Policy network
            value_model: Optional value network (can be same as policy)
            optimizer: Optimizer for models
            device: Device to run on
            
        Returns:
            Dictionary of training statistics
        """
        # Prepare batch data
        states = torch.stack([t['state'] for t in trajectories]).to(device)
        actions = torch.stack([t['action'] for t in trajectories]).to(device)
        rewards = torch.stack([t['reward'] for t in trajectories]).to(device)
        dones = torch.stack([t['done'] for t in trajectories]).to(device)
        old_log_probs = torch.stack([t['log_prob'] for t in trajectories]).to(device)
        
        # Compute V* for initial states
        v_star = self.compute_v_star(states[:, 0], policy_model, None, device)
        
        # Get value predictions if value model provided
        if value_model is not None:
            with torch.no_grad():
                values = value_model(states)['values']
        else:
            values = torch.zeros_like(rewards)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones, v_star)
        
        # Compute losses
        policy_loss = self.compute_policy_loss(
            states.view(-1, states.shape[-1]),
            actions.view(-1),
            advantages.view(-1),
            old_log_probs.view(-1),
            policy_model
        )
        
        if value_model is not None:
            value_loss = self.compute_value_loss(values.view(-1), returns.view(-1))
            total_loss = policy_loss + 0.5 * value_loss
        else:
            value_loss = torch.tensor(0.0)
            total_loss = policy_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), self.config.max_grad_norm)
        if value_model is not None:
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), self.config.max_grad_norm)
        
        optimizer.step()
        
        # Return statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if value_model is not None else 0.0,
            'total_loss': total_loss.item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'v_star_mean': v_star.mean().item() if v_star is not None else 0.0,
            'returns_mean': returns.mean().item()
        }
        
        return stats


class AStarPOTrainer:
    """
    High-level trainer class for A*PO
    Handles training loop and evaluation
    """
    
    def __init__(
        self,
        config: AStarPOConfig,
        policy_model: nn.Module,
        environment,
        device: torch.device = torch.device('cpu')
    ):
        self.config = config
        self.astarpo = AStarPO(config)
        self.policy_model = policy_model
        self.environment = environment
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.learning_rate
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = []
        
    def collect_trajectories(
        self,
        num_trajectories: int,
        max_steps: int = 100
    ) -> List[Dict]:
        """Collect trajectories using current policy"""
        trajectories = []
        
        for _ in range(num_trajectories):
            state = self.environment.reset()
            trajectory = []
            
            for step in range(max_steps):
                # Get action from policy
                with torch.no_grad():
                    state_tensor = self.encode_state(state)
                    outputs = self.policy_model(state_tensor)
                    action_logits = outputs['action_logits']
                    action_probs = F.softmax(action_logits, dim=-1)
                    action = torch.multinomial(action_probs, 1)
                    log_prob = torch.log(action_probs[0, action.item()])
                
                # Execute action
                next_state, reward, done, info = self.environment.step(action.item())
                
                # Store transition
                trajectory.append({
                    'state': state_tensor.squeeze(0),
                    'action': action.squeeze(),
                    'reward': torch.tensor(reward),
                    'done': torch.tensor(done),
                    'log_prob': log_prob
                })
                
                state = next_state
                
                if done:
                    break
            
            trajectories.extend(trajectory)
            
            # Record statistics
            episode_reward = sum([t['reward'].item() for t in trajectory])
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(trajectory))
        
        return trajectories
    
    def encode_state(self, state) -> torch.Tensor:
        """Encode environment state to tensor"""
        # This is a placeholder - implement based on your environment
        if isinstance(state, str):
            # Simple hash encoding for text states
            tokens = [hash(word) % 50000 for word in state.split()][:100]
            tokens += [0] * (100 - len(tokens))
            return torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        else:
            return torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def train(
        self,
        num_episodes: int,
        trajectories_per_update: int = 4,
        max_steps: int = 100
    ):
        """Main training loop"""
        print(f"Starting A*PO training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Collect trajectories
            trajectories = self.collect_trajectories(trajectories_per_update, max_steps)
            
            # Update policy using A*PO
            stats = self.astarpo.update(
                trajectories,
                self.policy_model,
                None,  # No separate value model in this implementation
                self.optimizer,
                self.device
            )
            
            self.training_stats.append(stats)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode:4d} | "
                      f"Reward: {avg_reward:7.3f} | "
                      f"Length: {avg_length:5.1f} | "
                      f"Loss: {stats['total_loss']:7.4f} | "
                      f"V*: {stats['v_star_mean']:7.3f}")
        
        print("Training completed!")
        return self.training_stats
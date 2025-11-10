"""
RAGEN Agent for WebShop and WebArena
Implements RAGEN with A*PO for web navigation
Compatible with both WebShop and WebArena environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from pathlib import Path

from models.transformer import WebNavigationModel
from algorithms.astarpo import AStarPO, AStarPOConfig
from environments.webshop_env import WebShopEnvironment, WebShopState


class WebShopRAGEN:
    """
    RAGEN Agent for WebShop and WebArena
    Uses A*PO for value estimation and policy optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize WebShop RAGEN agent"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model configuration
        model_config = config.get('model', {})
        self.hidden_dim = model_config.get('hidden_dim', 256)
        self.num_layers = model_config.get('num_layers', 4)
        self.num_heads = model_config.get('num_heads', 8)
        self.vocab_size = model_config.get('vocab_size', 50000)
        
        # Initialize model with larger action space for compatibility
        max_actions = max(
            len(WebShopEnvironment.ACTIONS),
            10  # Assuming WebArena has at most 10 actions
        )
        
        self.model = WebNavigationModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            action_vocab_size=max_actions,
            mlp_ratio=4.0,
            dropout=0.1
        ).to(self.device)
        
        # A*PO configuration
        astarpo_config = AStarPOConfig(
            beta=config.get('beta', 0.3),
            gamma=config.get('gamma', 0.95),
            num_samples=config.get('num_samples', 10),
            learning_rate=config.get('learning_rate', 1e-4)
        )
        self.astarpo = AStarPO(astarpo_config)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        
        # Training statistics
        self.episode_rewards = []
        self.success_rate = []
        self.training_steps = 0
        
        # State encoder
        self.max_text_length = 200
    
    def _detect_environment(self, state: Any) -> str:
        """Detect whether state is from WebShop or WebArena"""
        if hasattr(state, 'goal'):
            return 'webshop'
        elif hasattr(state, 'task') or hasattr(state, 'instruction'):
            return 'webarena'
        else:
            # Default to webshop if unclear
            return 'webshop'
    
    def _get_state_text(self, state: Any) -> str:
        """Extract text from state regardless of environment"""
        env_type = self._detect_environment(state)
        
        if env_type == 'webshop':
            # WebShop state
            if hasattr(state, 'to_text'):
                return state.to_text()
            else:
                goal = getattr(state, 'goal', '')
                obs = getattr(state, 'observation', {})
                return f"{goal} [SEP] {str(obs)}"
        else:
            # WebArena state
            task = getattr(state, 'task', getattr(state, 'instruction', ''))
            obs = getattr(state, 'observation', getattr(state, 'html', getattr(state, 'page_content', '')))
            return f"{task} [SEP] {str(obs)}"
    
    def encode_state(self, state: Any) -> torch.Tensor:
        """Encode state to tensor (works for both environments)"""
        # Convert state to text
        state_text = self._get_state_text(state)
        
        # Simple tokenization (in practice, use proper tokenizer)
        tokens = self._simple_tokenize(state_text)
        
        # Pad/truncate to fixed length
        if len(tokens) < self.max_text_length:
            tokens += [0] * (self.max_text_length - len(tokens))
        else:
            tokens = tokens[:self.max_text_length]
        
        return torch.tensor(tokens, dtype=torch.long, device=self.device)
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization by hashing words"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            # Remove punctuation
            word = ''.join(c for c in word if c.isalnum())
            if word:
                # Hash word to token ID
                token_id = hash(word) % (self.vocab_size - 1) + 1  # Reserve 0 for padding
                tokens.append(token_id)
        
        return tokens
    
    def _get_action_space_size(self, state: Any) -> int:
        """Get the number of actions for the current environment"""
        env_type = self._detect_environment(state)
        
        if env_type == 'webshop':
            return len(WebShopEnvironment.ACTIONS)
        else:
            # Import WebArena dynamically to avoid circular imports
            try:
                from environments.webarena_env import WebArenaEnvironment
                return len(WebArenaEnvironment.ACTIONS)
            except:
                # Default to a reasonable number
                return 10
    
    def select_action(
        self,
        state: Any,
        epsilon: float = 0.1
    ) -> Tuple[int, Optional[str]]:
        """
        Select action using RAGEN planning
        Works for both WebShop and WebArena states
        
        Returns:
            Tuple of (action_id, action_argument)
        """
        # Get action space size for current environment
        num_actions = self._get_action_space_size(state)
        
        # Encode state
        state_tensor = self.encode_state(state).unsqueeze(0)
        
        with torch.no_grad():
            # Get model predictions
            outputs = self.model.forward_rl(state_tensor)
            action_logits = outputs['action_logits']
            values = outputs['values']
            
            # Resize action logits if needed
            if action_logits.size(-1) > num_actions:
                action_logits = action_logits[:, :num_actions]
            elif action_logits.size(-1) < num_actions:
                # Pad with very negative values (low probability)
                padding = torch.full(
                    (action_logits.size(0), num_actions - action_logits.size(-1)),
                    -1e10,
                    device=self.device
                )
                action_logits = torch.cat([action_logits, padding], dim=1)
            
            # Apply A* search for planning
            if np.random.random() < epsilon:
                # Exploration
                if hasattr(state, 'get_available_actions'):
                    available_actions = state.get_available_actions()
                else:
                    available_actions = list(range(num_actions))
                action = np.random.choice(available_actions)
            else:
                # Exploitation with A* guidance
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
        
        # Generate action argument based on action type
        action_arg = self._generate_action_argument(state, action)
        
        return action, action_arg
    
    def _generate_action_argument(
        self,
        state: Any,
        action: int
    ) -> Optional[str]:
        """Generate argument for action (works for both environments)"""
        env_type = self._detect_environment(state)
        
        # Get appropriate actions dictionary
        if env_type == 'webshop':
            actions_dict = WebShopEnvironment.ACTIONS
        else:
            try:
                from environments.webarena_env import WebArenaEnvironment
                actions_dict = WebArenaEnvironment.ACTIONS
            except:
                return None
        
        # Get action name from action ID
        try:
            action_name = list(actions_dict.keys())[
                list(actions_dict.values()).index(action)
            ]
        except (IndexError, ValueError):
            return None
        
        # Get instruction text
        if env_type == 'webshop':
            instruction = getattr(state, 'goal', '')
        else:
            instruction = getattr(state, 'task', getattr(state, 'instruction', ''))
        
        if action_name == 'search':
            # Extract search terms from instruction
            instruction_words = instruction.lower().split()
            important_words = [w for w in instruction_words if len(w) > 3 and w not in 
                             ['under', 'less', 'than', 'find', 'buy', 'purchase',
                              'complete', 'task', 'need', 'want', 'please', 'navigate']]
            return ' '.join(important_words[:2]) if important_words else 'product'
        
        elif action_name == 'click':
            if env_type == 'webshop':
                # Click on first available item for WebShop
                if hasattr(state, 'search_results') and state.search_results:
                    return "0"  # Click first result
                elif hasattr(state, 'observation') and isinstance(state.observation, dict):
                    options = state.observation.get('options', [])
                    if options:
                        return str(options[0]) if options else "button"
                return "button"
            else:
                # WebArena click handling
                if hasattr(state, 'clickable_elements'):
                    elements = state.clickable_elements
                    if elements:
                        return str(elements[0])
                elif hasattr(state, 'links'):
                    links = state.links
                    if links:
                        return str(links[0])
                return "button"
        
        elif action_name == 'type':
            # Generate text based on context
            if 'search' in instruction.lower():
                words = instruction.lower().split()
                type_text = ' '.join([w for w in words if len(w) > 4][:2])
                return type_text if type_text else "example text"
            return "text"
        
        elif action_name == 'select':
            # Select dropdown options
            return "option_1"
        
        elif action_name in ['scroll', 'wait', 'back', 'forward', 'end']:
            # These actions typically don't need arguments
            return None
        
        # Default
        return None
    
    def compute_v_star_webshop(
        self,
        state: Any,
        env: Any,
        depth: int = 3
    ) -> float:
        """
        Compute V*(s) for WebShop/WebArena using lookahead search
        
        This is the key A*PO innovation: computing optimal value
        through search rather than learning
        """
        if depth == 0:
            return 0.0
        
        env_type = self._detect_environment(state)
        values = []
        
        # Sample multiple actions
        for _ in range(min(self.astarpo.config.num_samples, 5)):  # Limit samples for efficiency
            try:
                # Clone environment state (simplified - may need proper cloning based on env)
                if env_type == 'webshop':
                    env_copy = WebShopEnvironment(env.config)
                    env_copy.state = state
                    if hasattr(env, 'current_goal'):
                        env_copy.current_goal = env.current_goal
                else:
                    # For WebArena, we'll use a simpler approximation
                    env_copy = env
                
                # Simulate action
                action, action_arg = self.select_action(state, epsilon=0.3)
                next_state, reward, done, _ = env_copy.step(action, action_arg)
                
                if done:
                    value = reward
                else:
                    # Recursive value computation (reduced depth for efficiency)
                    future_value = self.astarpo.gamma * self.compute_v_star_webshop(
                        next_state, env_copy, depth - 1
                    )
                    value = reward + future_value
                
                values.append(value)
            except Exception as e:
                # If simulation fails, use a default value
                values.append(0.0)
        
        if not values:
            return 0.0
        
        # Compute V* using log-sum-exp
        values_tensor = torch.tensor(values, device=self.device)
        v_star = self.astarpo.beta * torch.logsumexp(
            values_tensor / self.astarpo.beta, dim=0
        )
        
        return v_star.item()
    
    def train_episode(
        self,
        env: Any
    ) -> Dict[str, float]:
        """Train one episode using RAGEN with A*PO"""
        state = env.reset()
        trajectory = []
        total_reward = 0.0
        
        for step in range(env.max_steps):
            # Select action
            action, action_arg = self.select_action(state, epsilon=0.2)
            
            # Encode current state
            state_tensor = self.encode_state(state)
            
            # Get action probability for policy gradient
            with torch.no_grad():
                outputs = self.model.forward_rl(state_tensor.unsqueeze(0))
                action_logits = outputs['action_logits']
                
                # Adjust for environment action space
                num_actions = self._get_action_space_size(state)
                if action_logits.size(-1) > num_actions:
                    action_logits = action_logits[:, :num_actions]
                
                action_probs = F.softmax(action_logits, dim=-1)
                log_prob = torch.log(action_probs[0, action] + 1e-8)
            
            # Execute action
            next_state, reward, done, info = env.step(action, action_arg)
            
            # Compute V* for advantage calculation (with reduced depth for speed)
            v_star = self.compute_v_star_webshop(state, env, depth=2)
            
            # Compute advantage A*(s,a) = r + γV*(s') - V*(s)
            if not done:
                v_star_next = self.compute_v_star_webshop(next_state, env, depth=2)
                advantage = reward + self.astarpo.gamma * v_star_next - v_star
            else:
                advantage = reward - v_star
            
            # Store transition
            trajectory.append({
                'state': state_tensor,
                'action': torch.tensor(action, device=self.device),
                'reward': torch.tensor(reward, device=self.device),
                'advantage': torch.tensor(advantage, device=self.device),
                'log_prob': log_prob,
                'done': torch.tensor(done, device=self.device)
            })
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update policy using trajectory
        self.update_policy(trajectory)
        
        # Record statistics
        success = info.get('success', False)
        self.episode_rewards.append(total_reward)
        self.success_rate.append(float(success))
        
        return {
            'reward': total_reward,
            'success': success,
            'steps': step + 1,
            'v_star_final': v_star
        }
    
    def update_policy(self, trajectory: List[Dict]) -> None:
        """Update policy using RAGEN objective with A*PO advantages"""
        if not trajectory:
            return
        
        # Stack trajectory data
        states = torch.stack([t['state'] for t in trajectory])
        actions = torch.stack([t['action'] for t in trajectory])
        advantages = torch.stack([t['advantage'] for t in trajectory])
        old_log_probs = torch.stack([t['log_prob'] for t in trajectory])
        
        # Forward pass
        outputs = self.model.forward_rl(states)
        action_logits = outputs['action_logits']
        values = outputs['values']
        
        # Get current log probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        current_log_probs = torch.log(
            action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8
        )
        
        # RAGEN policy loss with A*PO advantages
        # Use regression-style update for stability
        policy_loss = -(current_log_probs * advantages.detach()).mean()
        
        # KL penalty to prevent drastic changes
        kl_div = (old_log_probs - current_log_probs).mean()
        
        # Value loss (optional, for baseline)
        value_targets = torch.stack([t['reward'] for t in trajectory])
        value_loss = F.mse_loss(values.squeeze(), value_targets)
        
        # Total loss
        total_loss = policy_loss + self.astarpo.beta * kl_div + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
    
    def train(
        self,
        env: Any,
        num_episodes: int = 1000
    ) -> Dict[str, Any]:
        """Main training loop"""
        print(f"Starting RAGEN training for {num_episodes} episodes...")
        print("=" * 50)
        
        best_success_rate = 0.0
        
        for episode in range(num_episodes):
            # Train one episode
            stats = self.train_episode(env)
            
            # Log progress
            if episode % 50 == 0:
                recent_success = np.mean(self.success_rate[-50:]) if len(self.success_rate) >= 50 else np.mean(self.success_rate)
                recent_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
                
                print(f"Episode {episode:4d} | "
                      f"Success: {recent_success*100:5.1f}% | "
                      f"Reward: {recent_reward:6.3f} | "
                      f"Steps: {stats['steps']:3d} | "
                      f"V*: {stats['v_star_final']:6.3f}")
                
                if recent_success > best_success_rate:
                    best_success_rate = recent_success
                    self.save(f'checkpoints/best_model.pt')
                
                # Check for convergence
                if recent_success >= 0.80:
                    print(f"\n✅ Reached 80% success rate at episode {episode}!")
                    break
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print(f"Final Success Rate: {np.mean(self.success_rate[-100:])*100:.1f}%")
        print(f"Best Success Rate: {best_success_rate*100:.1f}%")
        
        return {
            'episode_rewards': self.episode_rewards,
            'success_rate': self.success_rate,
            'best_success_rate': best_success_rate,
            'final_success_rate': np.mean(self.success_rate[-100:])
        }
    
    def evaluate(
        self,
        env: Any,
        num_episodes: int = 100
    ) -> Dict[str, float]:
        """Evaluate agent performance"""
        print(f"Evaluating on {num_episodes} episodes...")
        
        eval_rewards = []
        eval_successes = []
        eval_steps = []
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            
            for step in range(env.max_steps):
                # Select action (no exploration)
                action, action_arg = self.select_action(state, epsilon=0.0)
                state, reward, done, info = env.step(action, action_arg)
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_successes.append(info.get('success', False))
            eval_steps.append(step + 1)
        
        results = {
            'success_rate': np.mean(eval_successes) * 100,
            'avg_reward': np.mean(eval_rewards),
            'avg_steps': np.mean(eval_steps),
            'std_reward': np.std(eval_rewards),
            'min_steps': min(eval_steps) if eval_steps else 0,
            'max_steps': max(eval_steps) if eval_steps else 0
        }
        
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Avg Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"Avg Steps: {results['avg_steps']:.1f} ({results['min_steps']}-{results['max_steps']})")
        
        return results
    
    def save(self, path: str) -> None:
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'success_rate': self.success_rate,
            'training_steps': self.training_steps
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.success_rate = checkpoint.get('success_rate', [])
        self.training_steps = checkpoint.get('training_steps', 0)
        
        print(f"Model loaded from {path}")
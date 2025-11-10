"""
Baseline agents for comparison with RAGEN
Includes Random, Rule-based, and simple heuristic agents
"""

import random
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all agents"""
    
    @abstractmethod
    def act(self, state: Any) -> int:
        """Select action given state"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset agent for new episode"""
        pass


class RandomAgent(BaseAgent):
    """
    Random baseline agent
    Selects actions uniformly at random
    """
    
    def __init__(self, action_space: Dict[str, int]):
        """Initialize random agent"""
        self.action_space = action_space
        self.action_list = list(action_space.values())
        
    def act(self, state: Any) -> int:
        """Select random action"""
        return random.choice(self.action_list)
    
    def reset(self):
        """Reset agent (nothing to reset for random)"""
        pass


class RuleBasedAgent(BaseAgent):
    """
    Rule-based agent for WebShop/WebArena
    Uses hand-crafted rules based on state
    """
    
    def __init__(self, rules_config: Dict[str, Any] = None):
        """Initialize rule-based agent"""
        self.rules_config = rules_config or {}
        self.visited_states = set()
        self.search_performed = False
        self.item_clicked = False
        self.current_step = 0
        
        # Define action mappings
        self.actions = {
            'search': 0,
            'click': 1,
            'buy': 2,
            'back': 3,
            'next_page': 4,
            'prev_page': 5
        }
        
    def act(self, state: Any) -> int:
        """Select action based on rules"""
        self.current_step += 1
        
        # Convert state to string for pattern matching
        if hasattr(state, 'to_text'):
            state_text = state.to_text()
        else:
            state_text = str(state)
        
        # Apply rules based on state
        if '[Search Page]' in state_text and not self.search_performed:
            # On search page and haven't searched yet
            self.search_performed = True
            return self.actions['search']
            
        elif '[Results' in state_text and not self.item_clicked:
            # On results page, click first item
            self.item_clicked = True
            return self.actions['click']
            
        elif '[Product Page]' in state_text:
            # On product page, check price and buy if reasonable
            if self._check_price_in_budget(state_text):
                return self.actions['buy']
            else:
                return self.actions['back']
                
        elif '[Success]' in state_text:
            # Task completed
            return self.actions['back']  # No-op
            
        else:
            # Default: go back or search
            if self.current_step > 5:
                return self.actions['back']
            return self.actions['search']
    
    def _check_price_in_budget(self, state_text: str) -> bool:
        """Check if price is within budget"""
        # Extract price from state
        price_match = re.search(r'\$(\d+\.?\d*)', state_text)
        if price_match:
            price = float(price_match.group(1))
            
            # Extract budget from state
            budget_match = re.search(r'Budget: \$(\d+\.?\d*)', state_text)
            if budget_match:
                budget = float(budget_match.group(1))
                return price <= budget
        
        return False
    
    def reset(self):
        """Reset agent for new episode"""
        self.visited_states = set()
        self.search_performed = False
        self.item_clicked = False
        self.current_step = 0


class HeuristicAgent(BaseAgent):
    """
    Heuristic-based agent with simple planning
    Better than random but simpler than RAGEN
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize heuristic agent"""
        self.config = config or {}
        self.plan = []
        self.plan_index = 0
        self.state_history = []
        self.information_buffer = {}
        
        self.actions = {
            'search': 0,
            'click': 1,
            'buy': 2,
            'back': 3,
            'next_page': 4,
            'prev_page': 5
        }
        
    def act(self, state: Any) -> int:
        """Select action using heuristics"""
        # Store state history
        self.state_history.append(state)
        
        # Generate plan if empty
        if not self.plan or self.plan_index >= len(self.plan):
            self.plan = self._generate_plan(state)
            self.plan_index = 0
        
        # Execute next action in plan
        if self.plan_index < len(self.plan):
            action = self.plan[self.plan_index]
            self.plan_index += 1
            return action
        
        # Fallback to greedy action
        return self._greedy_action(state)
    
    def _generate_plan(self, state: Any) -> List[int]:
        """Generate simple plan based on current state"""
        plan = []
        
        if hasattr(state, 'to_text'):
            state_text = state.to_text()
        else:
            state_text = str(state)
        
        # Simple planning based on state
        if '[Search Page]' in state_text:
            # Plan: Search -> Click -> Buy
            plan = [
                self.actions['search'],
                self.actions['click'],
                self.actions['buy']
            ]
        elif '[Results' in state_text:
            # Plan: Click -> Buy
            plan = [
                self.actions['click'],
                self.actions['buy']
            ]
        elif '[Product Page]' in state_text:
            # Plan: Buy or Back
            if self._is_good_product(state):
                plan = [self.actions['buy']]
            else:
                plan = [self.actions['back'], self.actions['click']]
        
        return plan
    
    def _greedy_action(self, state: Any) -> int:
        """Select greedy action based on immediate reward expectation"""
        if hasattr(state, 'page_type'):
            page_type = state.page_type
            
            if page_type == 'search':
                return self.actions['search']
            elif page_type == 'results':
                return self.actions['click']
            elif page_type == 'item':
                return self.actions['buy']
        
        # Default
        return self.actions['search']
    
    def _is_good_product(self, state: Any) -> bool:
        """Check if current product meets criteria"""
        if hasattr(state, 'current_product'):
            product = state.current_product
            if hasattr(product, 'price') and hasattr(state, 'budget'):
                return product.price <= state.budget
        
        return False
    
    def reset(self):
        """Reset agent for new episode"""
        self.plan = []
        self.plan_index = 0
        self.state_history = []
        self.information_buffer = {}


class ReactiveAgent(BaseAgent):
    """
    Reactive agent that responds to immediate state
    No memory or planning
    """
    
    def __init__(self):
        """Initialize reactive agent"""
        self.action_map = {
            'search': 0,
            'results': 1,  # Click
            'item': 2,     # Buy
            'cart': 2,     # Buy
            'success': 3,  # Back
            'default': 0   # Search
        }
        
    def act(self, state: Any) -> int:
        """React to immediate state"""
        # Identify state type
        state_type = self._identify_state_type(state)
        
        # Map state to action
        return self.action_map.get(state_type, self.action_map['default'])
    
    def _identify_state_type(self, state: Any) -> str:
        """Identify type of current state"""
        if hasattr(state, 'page_type'):
            return state.page_type
        
        # Parse from text
        state_text = str(state)
        if 'Search' in state_text:
            return 'search'
        elif 'Results' in state_text:
            return 'results'
        elif 'Product' in state_text or 'Item' in state_text:
            return 'item'
        elif 'Cart' in state_text:
            return 'cart'
        elif 'Success' in state_text:
            return 'success'
        
        return 'default'
    
    def reset(self):
        """Reset agent (nothing to reset)"""
        pass


class GreedyAgent(BaseAgent):
    """
    Greedy agent that always selects action with highest immediate reward
    """
    
    def __init__(self, reward_estimates: Dict[str, float] = None):
        """Initialize greedy agent"""
        self.reward_estimates = reward_estimates or {
            'search': 0.1,
            'click': 0.2,
            'buy': 1.0,
            'back': 0.0,
            'next_page': 0.05,
            'prev_page': 0.05
        }
        
        self.actions = {
            'search': 0,
            'click': 1,
            'buy': 2,
            'back': 3,
            'next_page': 4,
            'prev_page': 5
        }
        
    def act(self, state: Any) -> int:
        """Select action with highest expected reward"""
        # Get available actions for current state
        available_actions = self._get_available_actions(state)
        
        if not available_actions:
            return self.actions['back']  # Default
        
        # Select action with highest reward estimate
        best_action = None
        best_reward = -float('inf')
        
        for action_name in available_actions:
            reward = self.reward_estimates.get(action_name, 0.0)
            if reward > best_reward:
                best_reward = reward
                best_action = action_name
        
        return self.actions.get(best_action, 0)
    
    def _get_available_actions(self, state: Any) -> List[str]:
        """Get available actions for current state"""
        if hasattr(state, 'page_type'):
            page_type = state.page_type
            
            if page_type == 'search':
                return ['search']
            elif page_type == 'results':
                return ['click', 'back', 'next_page']
            elif page_type == 'item':
                return ['buy', 'back']
            elif page_type == 'cart':
                return ['buy', 'back']
        
        return ['search', 'back']
    
    def reset(self):
        """Reset agent"""
        pass


def create_baseline_agent(agent_type: str, config: Dict[str, Any] = None) -> BaseAgent:
    """
    Factory function to create baseline agents
    
    Args:
        agent_type: Type of agent ('random', 'rule_based', 'heuristic', 'reactive', 'greedy')
        config: Configuration for the agent
        
    Returns:
        BaseAgent instance
    """
    if agent_type == 'random':
        action_space = config.get('action_space', {
            'search': 0, 'click': 1, 'buy': 2,
            'back': 3, 'next_page': 4, 'prev_page': 5
        })
        return RandomAgent(action_space)
        
    elif agent_type == 'rule_based':
        return RuleBasedAgent(config)
        
    elif agent_type == 'heuristic':
        return HeuristicAgent(config)
        
    elif agent_type == 'reactive':
        return ReactiveAgent()
        
    elif agent_type == 'greedy':
        return GreedyAgent(config.get('reward_estimates'))
        
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_baseline_agents(env, num_episodes: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline agents on the environment
    
    Args:
        env: Environment to evaluate on
        num_episodes: Number of episodes per agent
        
    Returns:
        Dictionary with results for each agent
    """
    results = {}
    
    agent_types = ['random', 'rule_based', 'heuristic', 'reactive', 'greedy']
    
    for agent_type in agent_types:
        print(f"Evaluating {agent_type} agent...")
        
        agent = create_baseline_agent(agent_type)
        
        episode_rewards = []
        episode_successes = []
        episode_steps = []
        
        for _ in range(num_episodes):
            state = env.reset()
            agent.reset()
            
            total_reward = 0
            steps = 0
            
            for _ in range(env.max_steps):
                action = agent.act(state)
                state, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_successes.append(info.get('success', False))
            episode_steps.append(steps)
        
        results[agent_type] = {
            'success_rate': np.mean(episode_successes) * 100,
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_steps': np.mean(episode_steps),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        print(f"  Success Rate: {results[agent_type]['success_rate']:.1f}%")
        print(f"  Avg Reward: {results[agent_type]['avg_reward']:.3f}")
    
    return results
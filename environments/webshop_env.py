"""
WebShop Environment
Simulates an e-commerce website for product search and purchase
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Product:
    """Product in the WebShop"""
    id: str
    title: str
    price: float
    features: List[str]
    rating: float
    category: str
    description: str
    available: bool = True
    
    def matches_criteria(self, criteria: List[str]) -> float:
        """Calculate how well product matches search criteria"""
        score = 0.0
        text = f"{self.title} {self.description} {' '.join(self.features)}".lower()
        
        for criterion in criteria:
            if criterion.lower() in text:
                score += 1.0
                
        return score / len(criteria) if criteria else 0.0


@dataclass
class WebShopState:
    """Current state of the WebShop environment"""
    page_type: str  # 'search', 'results', 'item', 'cart', 'success'
    current_product: Optional[Product] = None
    search_results: List[Product] = None
    search_query: str = ""
    cart: List[Product] = None
    budget: float = 100.0
    goal: str = ""
    step_count: int = 0
    
    def __post_init__(self):
        if self.search_results is None:
            self.search_results = []
        if self.cart is None:
            self.cart = []
    
    def to_text(self) -> str:
        """Convert state to text representation"""
        if self.page_type == 'search':
            return f"[Search Page] Goal: {self.goal} | Budget: ${self.budget:.2f}"
        elif self.page_type == 'results':
            results_text = f"[Results for '{self.search_query}'] Found {len(self.search_results)} products\n"
            for i, product in enumerate(self.search_results[:5]):
                results_text += f"{i+1}. {product.title} - ${product.price:.2f}\n"
            return results_text
        elif self.page_type == 'item':
            if self.current_product:
                return f"[Product Page] {self.current_product.title}\nPrice: ${self.current_product.price:.2f}\nRating: {self.current_product.rating}/5\nFeatures: {', '.join(self.current_product.features[:3])}"
            return "[Product Page] No product selected"
        elif self.page_type == 'cart':
            cart_text = f"[Shopping Cart] {len(self.cart)} items\n"
            total = sum(p.price for p in self.cart)
            cart_text += f"Total: ${total:.2f} | Budget: ${self.budget:.2f}"
            return cart_text
        elif self.page_type == 'success':
            return "[Success] Purchase completed!"
        else:
            return f"[Unknown Page] {self.page_type}"


class WebShopEnvironment:
    """
    WebShop Environment for training web navigation agents
    Based on the WebShop benchmark paper
    """
    
    # Action definitions
    ACTIONS = {
        'search': 0,      # Perform search with query
        'click': 1,       # Click on an element
        'buy': 2,         # Purchase current item
        'back': 3,        # Go back to previous page
        'next_page': 4,   # Go to next results page
        'prev_page': 5,   # Go to previous results page
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize WebShop environment"""
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 10)
        self.num_products = self.config.get('num_products', 1000)
        self.reward_success = self.config.get('reward_success', 1.0)
        self.reward_partial = self.config.get('reward_partial', 0.5)
        self.reward_step = self.config.get('reward_step', -0.01)
        
        # Load or generate products
        self.products = self._load_products()
        
        # Shopping goals
        self.goals = self._load_goals()
        
        # Initialize state
        self.reset()
        
    def _load_products(self) -> List[Product]:
        """Load product database"""
        products = []
        
        # Try to load from file
        data_file = Path('data/webshop/products.json')
        if data_file.exists():
            with open(data_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    products.append(Product(**item))
        else:
            # Generate synthetic products
            categories = ['electronics', 'home', 'clothing', 'books', 'sports']
            
            for i in range(self.num_products):
                category = random.choice(categories)
                products.append(self._generate_product(f"P{i:04d}", category))
        
        return products
    
    def _generate_product(self, product_id: str, category: str) -> Product:
        """Generate a synthetic product"""
        templates = {
            'electronics': {
                'titles': ['Wireless Headphones', 'Smartphone', 'Laptop', 'Tablet', 'Smart Watch'],
                'features': ['Bluetooth', 'Wireless', 'Fast charging', 'HD display', 'Long battery'],
                'price_range': (50, 500)
            },
            'home': {
                'titles': ['Coffee Maker', 'Vacuum Cleaner', 'Air Purifier', 'Blender', 'Toaster'],
                'features': ['Energy efficient', 'Easy to clean', 'Compact', 'Modern design', 'Durable'],
                'price_range': (20, 200)
            },
            'clothing': {
                'titles': ['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Shoes'],
                'features': ['Cotton', 'Comfortable', 'Machine washable', 'All sizes', 'Multiple colors'],
                'price_range': (15, 150)
            },
            'books': {
                'titles': ['Novel', 'Cookbook', 'Textbook', 'Biography', 'Self-help Book'],
                'features': ['Bestseller', 'Hardcover', 'Illustrated', 'Award winning', 'New release'],
                'price_range': (10, 50)
            },
            'sports': {
                'titles': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Gym Bag', 'Fitness Tracker'],
                'features': ['Non-slip', 'Durable', 'Lightweight', 'Waterproof', 'Adjustable'],
                'price_range': (20, 150)
            }
        }
        
        template = templates[category]
        title = random.choice(template['titles'])
        features = random.sample(template['features'], 3)
        price = random.uniform(*template['price_range'])
        
        return Product(
            id=product_id,
            title=f"{title} {product_id}",
            price=round(price, 2),
            features=features,
            rating=round(random.uniform(3.5, 5.0), 1),
            category=category,
            description=f"High quality {title.lower()} with {', '.join(features[:2])}",
            available=True
        )
    
    def _load_goals(self) -> List[Dict[str, Any]]:
        """Load shopping goals"""
        goals = [
            {
                'description': "Buy wireless headphones under $100",
                'target_features': ['wireless', 'headphones'],
                'budget': 100.0,
                'category': 'electronics'
            },
            {
                'description': "Find a coffee maker under $50",
                'target_features': ['coffee', 'maker'],
                'budget': 50.0,
                'category': 'home'
            },
            {
                'description': "Purchase running shoes under $80",
                'target_features': ['running', 'shoes'],
                'budget': 80.0,
                'category': 'sports'
            },
            {
                'description': "Get a bestseller book under $30",
                'target_features': ['bestseller', 'book'],
                'budget': 30.0,
                'category': 'books'
            },
            {
                'description': "Buy a comfortable t-shirt under $40",
                'target_features': ['comfortable', 't-shirt'],
                'budget': 40.0,
                'category': 'clothing'
            }
        ]
        return goals
    
    def reset(self, goal_index: Optional[int] = None) -> WebShopState:
        """Reset environment with new goal"""
        # Select goal
        if goal_index is not None and 0 <= goal_index < len(self.goals):
            self.current_goal = self.goals[goal_index]
        else:
            self.current_goal = random.choice(self.goals)
        
        # Initialize state
        self.state = WebShopState(
            page_type='search',
            goal=self.current_goal['description'],
            budget=self.current_goal['budget'],
            step_count=0
        )
        
        self.done = False
        self.total_reward = 0.0
        
        return self.state
    
    def get_available_actions(self) -> List[int]:
        """Get list of valid actions in current state"""
        actions = []
        
        if self.state.page_type == 'search':
            actions = [self.ACTIONS['search']]
        elif self.state.page_type == 'results':
            actions = [self.ACTIONS['click'], self.ACTIONS['back']]
            if len(self.state.search_results) > 5:
                actions.extend([self.ACTIONS['next_page'], self.ACTIONS['prev_page']])
        elif self.state.page_type == 'item':
            actions = [self.ACTIONS['buy'], self.ACTIONS['back']]
        elif self.state.page_type == 'cart':
            actions = [self.ACTIONS['buy'], self.ACTIONS['back']]
            
        return actions
    
    def step(self, action: int, action_arg: Optional[str] = None) -> Tuple[WebShopState, float, bool, Dict]:
        """
        Execute action in environment
        
        Args:
            action: Action ID from ACTIONS
            action_arg: Optional argument for action (e.g., search query, item index)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            return self.state, 0.0, True, {'message': 'Episode already done'}
        
        self.state.step_count += 1
        reward = self.reward_step  # Small negative reward per step
        info = {}
        
        # Execute action based on current state and action type
        if action == self.ACTIONS['search'] and self.state.page_type == 'search':
            reward += self._execute_search(action_arg)
            
        elif action == self.ACTIONS['click'] and self.state.page_type == 'results':
            reward += self._execute_click(action_arg)
            
        elif action == self.ACTIONS['buy']:
            reward += self._execute_buy()
            
        elif action == self.ACTIONS['back']:
            reward += self._execute_back()
            
        else:
            info['message'] = 'Invalid action for current state'
            reward -= 0.1  # Penalty for invalid action
        
        # Check termination conditions
        if self.state.page_type == 'success':
            self.done = True
            info['success'] = True
        elif self.state.step_count >= self.max_steps:
            self.done = True
            info['success'] = False
            reward -= 0.5  # Penalty for timeout
        
        self.total_reward += reward
        info['total_reward'] = self.total_reward
        
        return self.state, reward, self.done, info
    
    def _execute_search(self, query: Optional[str]) -> float:
        """Execute search action"""
        if query is None:
            # Generate query from goal
            query = ' '.join(self.current_goal['target_features'])
        
        self.state.search_query = query
        
        # Find matching products
        results = []
        query_words = query.lower().split()
        
        for product in self.products:
            score = product.matches_criteria(query_words)
            if score > 0:
                results.append((product, score))
        
        # Sort by relevance and price
        results.sort(key=lambda x: (x[1], -x[0].price), reverse=True)
        self.state.search_results = [r[0] for r in results[:10]]
        
        self.state.page_type = 'results'
        
        # Reward for finding relevant results
        if self.state.search_results:
            return 0.1
        return 0.0
    
    def _execute_click(self, item_index: Optional[str]) -> float:
        """Execute click action on search result"""
        if item_index is None:
            return -0.1  # Penalty for no item specified
        
        try:
            index = int(item_index)
            if 0 <= index < len(self.state.search_results):
                self.state.current_product = self.state.search_results[index]
                self.state.page_type = 'item'
                
                # Reward for selecting relevant item
                if self.state.current_product.price <= self.state.budget:
                    return 0.2
                return 0.1
        except (ValueError, IndexError):
            pass
        
        return -0.1  # Penalty for invalid index
    
    def _execute_buy(self) -> float:
        """Execute buy action"""
        if self.state.page_type == 'item' and self.state.current_product:
            product = self.state.current_product
            
            # Check budget constraint
            if product.price > self.state.budget:
                return -0.3  # Penalty for exceeding budget
            
            # Check if product matches goal
            match_score = product.matches_criteria(self.current_goal['target_features'])
            
            if match_score > 0.5:
                # Success!
                self.state.page_type = 'success'
                self.state.cart.append(product)
                return self.reward_success
            else:
                # Partial success
                self.state.page_type = 'success'
                self.state.cart.append(product)
                return self.reward_partial * match_score
        
        return -0.2  # Penalty for buying without product
    
    def _execute_back(self) -> float:
        """Execute back navigation"""
        if self.state.page_type == 'item':
            self.state.page_type = 'results'
            self.state.current_product = None
        elif self.state.page_type == 'results':
            self.state.page_type = 'search'
            self.state.search_results = []
        
        return 0.0  # No reward for navigation
    
    def render(self, mode: str = 'text') -> str:
        """Render current state"""
        return self.state.to_text()
    
    def get_state_vector(self) -> np.ndarray:
        """Convert state to vector representation"""
        # Simple encoding - in practice would use embeddings
        state_features = [
            float(self.state.page_type == 'search'),
            float(self.state.page_type == 'results'),
            float(self.state.page_type == 'item'),
            float(self.state.page_type == 'success'),
            len(self.state.search_results) / 10.0,
            len(self.state.cart) / 5.0,
            self.state.budget / 100.0,
            self.state.step_count / self.max_steps,
        ]
        
        return np.array(state_features, dtype=np.float32)
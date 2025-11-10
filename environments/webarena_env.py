"""
WebArena Environment
Complex multi-site web navigation tasks
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class WebSite(Enum):
    """Different websites in WebArena"""
    SHOPPING = "shopping"
    SOCIAL = "social"
    PRODUCTIVITY = "productivity"
    BANKING = "banking"
    TRAVEL = "travel"
    NEWS = "news"
    EMAIL = "email"


@dataclass
class WebArenaTask:
    """Complex multi-step task in WebArena"""
    id: str
    name: str
    description: str
    sites_involved: List[WebSite]
    steps_required: int
    subtasks: List[str]
    success_criteria: Dict[str, Any]
    complexity: str  # 'easy', 'medium', 'hard'
    timeout: int = 30


@dataclass
class WebArenaState:
    """State representation for WebArena"""
    current_site: WebSite
    current_page: str
    page_content: Dict[str, Any]
    task_progress: Dict[str, bool]  # Subtask completion status
    information_gathered: Dict[str, Any]  # Cross-site information
    step_count: int
    task: WebArenaTask
    history: List[str] = field(default_factory=list)
    
    def to_text(self) -> str:
        """Convert state to text representation"""
        progress = sum(self.task_progress.values()) / len(self.task_progress) * 100
        return (f"[{self.current_site.value}:{self.current_page}] "
                f"Task: {self.task.name} | "
                f"Progress: {progress:.0f}% | "
                f"Steps: {self.step_count}/{self.task.timeout}")
    
    def is_complete(self) -> bool:
        """Check if all subtasks are complete"""
        return all(self.task_progress.values())


class WebArenaEnvironment:
    """
    WebArena Environment for complex web navigation
    Simulates multi-site interactions and complex tasks
    """
    
    # Extended action space
    ACTIONS = {
        # Navigation
        'click': 0,
        'type': 1,
        'submit': 2,
        'back': 3,
        'switch_tab': 4,
        'open_new_tab': 5,
        'close_tab': 6,
        
        # Page interactions
        'scroll_down': 7,
        'scroll_up': 8,
        'select_dropdown': 9,
        'checkbox': 10,
        'radio_button': 11,
        
        # Data operations
        'copy': 12,
        'paste': 13,
        'save_info': 14,
        'compare': 15,
        
        # Site-specific
        'login': 16,
        'logout': 17,
        'search': 18,
        'filter': 19,
        'sort': 20,
        
        # Complex actions
        'extract_table': 21,
        'fill_form': 22,
        'download': 23,
        'upload': 24,
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize WebArena environment"""
        self.config = config or {}
        self.max_steps = self.config.get('max_steps', 30)
        
        # Load tasks
        self.tasks = self._load_tasks()
        
        # Site simulators
        self.sites = self._initialize_sites()
        
        # Initialize state
        self.reset()
    
    def _load_tasks(self) -> List[WebArenaTask]:
        """Load WebArena tasks"""
        tasks = []
        
        # Try to load from file
        data_file = Path('data/webarena/tasks.json')
        if data_file.exists():
            with open(data_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    tasks.append(WebArenaTask(**item))
        else:
            # Create sample tasks
            tasks = [
                WebArenaTask(
                    id="WA001",
                    name="Multi-Site Shopping Comparison",
                    description="Compare prices for a laptop across 3 sites and buy from cheapest",
                    sites_involved=[WebSite.SHOPPING, WebSite.NEWS],
                    steps_required=15,
                    subtasks=[
                        "Search for laptop on site 1",
                        "Note price and specs",
                        "Search on site 2",
                        "Compare prices",
                        "Purchase from cheapest"
                    ],
                    success_criteria={"purchased": True, "best_price": True},
                    complexity="medium"
                ),
                WebArenaTask(
                    id="WA002",
                    name="Travel Planning",
                    description="Book flight and hotel for business trip with calendar coordination",
                    sites_involved=[WebSite.TRAVEL, WebSite.EMAIL, WebSite.PRODUCTIVITY],
                    steps_required=20,
                    subtasks=[
                        "Check calendar for available dates",
                        "Search flights",
                        "Compare flight options",
                        "Search hotels near destination",
                        "Book flight and hotel",
                        "Add to calendar",
                        "Send confirmation email"
                    ],
                    success_criteria={"flight_booked": True, "hotel_booked": True, "calendar_updated": True},
                    complexity="hard"
                ),
                WebArenaTask(
                    id="WA003",
                    name="Social Media Research",
                    description="Research trending topics and create summary report",
                    sites_involved=[WebSite.SOCIAL, WebSite.NEWS, WebSite.PRODUCTIVITY],
                    steps_required=12,
                    subtasks=[
                        "Check trending topics on social",
                        "Search news articles",
                        "Extract key information",
                        "Create document",
                        "Write summary"
                    ],
                    success_criteria={"report_created": True, "sources_cited": 3},
                    complexity="medium"
                ),
                WebArenaTask(
                    id="WA004",
                    name="Account Management",
                    description="Update payment methods across multiple services",
                    sites_involved=[WebSite.BANKING, WebSite.SHOPPING, WebSite.PRODUCTIVITY],
                    steps_required=18,
                    subtasks=[
                        "Login to banking",
                        "Get new card details",
                        "Update shopping site payment",
                        "Update subscription payment",
                        "Verify all updates"
                    ],
                    success_criteria={"all_updated": True, "verified": True},
                    complexity="hard"
                ),
                WebArenaTask(
                    id="WA005",
                    name="Email Campaign",
                    description="Create and send targeted email campaign",
                    sites_involved=[WebSite.EMAIL, WebSite.PRODUCTIVITY],
                    steps_required=10,
                    subtasks=[
                        "Create email template",
                        "Import contact list",
                        "Personalize content",
                        "Schedule sending",
                        "Track opens"
                    ],
                    success_criteria={"emails_sent": True, "scheduled": True},
                    complexity="easy"
                )
            ]
        
        return tasks
    
    def _initialize_sites(self) -> Dict[WebSite, Dict]:
        """Initialize website simulators"""
        sites = {}
        
        for site in WebSite:
            sites[site] = {
                'pages': self._generate_site_pages(site),
                'data': self._generate_site_data(site),
                'logged_in': False
            }
        
        return sites
    
    def _generate_site_pages(self, site: WebSite) -> Dict[str, Dict]:
        """Generate pages for a website"""
        pages = {
            'home': {'title': f'{site.value} Home', 'elements': ['search', 'login', 'menu']},
            'search': {'title': 'Search Results', 'elements': ['results', 'filters', 'sort']},
            'product': {'title': 'Product Details', 'elements': ['price', 'specs', 'buy_button']},
            'account': {'title': 'Account Settings', 'elements': ['profile', 'payment', 'security']},
        }
        return pages
    
    def _generate_site_data(self, site: WebSite) -> Dict[str, Any]:
        """Generate data for a website"""
        if site == WebSite.SHOPPING:
            return {
                'products': [
                    {'id': 'L1', 'name': 'Laptop Pro', 'price': 999},
                    {'id': 'L2', 'name': 'Laptop Air', 'price': 799},
                    {'id': 'L3', 'name': 'Laptop Basic', 'price': 599}
                ]
            }
        elif site == WebSite.TRAVEL:
            return {
                'flights': [
                    {'id': 'F1', 'from': 'NYC', 'to': 'LAX', 'price': 299},
                    {'id': 'F2', 'from': 'NYC', 'to': 'LAX', 'price': 399}
                ],
                'hotels': [
                    {'id': 'H1', 'name': 'Grand Hotel', 'price': 199},
                    {'id': 'H2', 'name': 'Business Inn', 'price': 99}
                ]
            }
        else:
            return {}
    
    def reset(self, task_id: Optional[str] = None) -> WebArenaState:
        """Reset environment with specified or random task"""
        # Select task
        if task_id:
            self.current_task = next((t for t in self.tasks if t.id == task_id), self.tasks[0])
        else:
            self.current_task = random.choice(self.tasks)
        
        # Initialize state
        self.state = WebArenaState(
            current_site=self.current_task.sites_involved[0],
            current_page='home',
            page_content=self.sites[self.current_task.sites_involved[0]]['pages']['home'],
            task_progress={subtask: False for subtask in self.current_task.subtasks},
            information_gathered={},
            step_count=0,
            task=self.current_task
        )
        
        self.done = False
        self.total_reward = 0.0
        
        # Reset site states
        for site_data in self.sites.values():
            site_data['logged_in'] = False
        
        return self.state
    
    def step(self, action: int, action_params: Dict[str, Any] = None) -> Tuple[WebArenaState, float, bool, Dict]:
        """
        Execute action in environment
        
        Args:
            action: Action ID from ACTIONS
            action_params: Parameters for the action
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            return self.state, 0.0, True, {'message': 'Episode already done'}
        
        self.state.step_count += 1
        reward = -0.01  # Small negative reward per step
        info = {}
        
        # Record action in history
        action_name = list(self.ACTIONS.keys())[list(self.ACTIONS.values()).index(action)]
        self.state.history.append(f"{action_name}: {action_params}")
        
        # Execute action
        if action == self.ACTIONS['click']:
            reward += self._execute_click(action_params)
        elif action == self.ACTIONS['type']:
            reward += self._execute_type(action_params)
        elif action == self.ACTIONS['search']:
            reward += self._execute_search(action_params)
        elif action == self.ACTIONS['switch_tab']:
            reward += self._execute_switch_tab(action_params)
        elif action == self.ACTIONS['save_info']:
            reward += self._execute_save_info(action_params)
        elif action == self.ACTIONS['compare']:
            reward += self._execute_compare()
        elif action == self.ACTIONS['submit']:
            reward += self._execute_submit()
        else:
            info['message'] = f'Action {action_name} not fully implemented'
        
        # Check subtask completion
        self._update_task_progress()
        
        # Calculate reward based on progress
        progress_reward = self._calculate_progress_reward()
        reward += progress_reward
        
        # Check termination
        if self.state.is_complete():
            self.done = True
            reward += 10.0  # Large reward for task completion
            info['success'] = True
        elif self.state.step_count >= self.current_task.timeout:
            self.done = True
            reward -= 5.0  # Penalty for timeout
            info['success'] = False
        
        self.total_reward += reward
        info['total_reward'] = self.total_reward
        info['progress'] = sum(self.state.task_progress.values()) / len(self.state.task_progress)
        
        return self.state, reward, self.done, info
    
    def _execute_click(self, params: Dict) -> float:
        """Execute click action"""
        if not params or 'element' not in params:
            return -0.1
        
        element = params['element']
        current_page = self.sites[self.state.current_site]['pages'].get(self.state.current_page, {})
        
        if element in current_page.get('elements', []):
            # Navigate to new page based on element
            if element == 'search':
                self.state.current_page = 'search'
                return 0.1
            elif element == 'login':
                self.sites[self.state.current_site]['logged_in'] = True
                return 0.2
            elif element == 'buy_button':
                # Check if we have compared prices
                if 'best_price' in self.state.information_gathered:
                    return 1.0  # Reward for smart purchase
                return 0.3
        
        return 0.0
    
    def _execute_type(self, params: Dict) -> float:
        """Execute type action"""
        if not params or 'text' not in params:
            return -0.1
        
        # Store typed text for later use
        self.state.information_gathered['last_typed'] = params['text']
        return 0.05
    
    def _execute_search(self, params: Dict) -> float:
        """Execute search action"""
        if 'query' not in params:
            return -0.1
        
        query = params['query']
        site_data = self.sites[self.state.current_site]['data']
        
        # Simulate search results
        if self.state.current_site == WebSite.SHOPPING:
            products = site_data.get('products', [])
            results = [p for p in products if query.lower() in p['name'].lower()]
            self.state.information_gathered['search_results'] = results
            
            if results:
                self.state.current_page = 'search'
                return 0.2
        
        return 0.1
    
    def _execute_switch_tab(self, params: Dict) -> float:
        """Switch to different website"""
        if 'site' not in params:
            return -0.1
        
        target_site = params['site']
        if isinstance(target_site, str):
            # Convert string to WebSite enum
            try:
                target_site = WebSite(target_site)
            except ValueError:
                return -0.1
        
        if target_site in self.current_task.sites_involved:
            self.state.current_site = target_site
            self.state.current_page = 'home'
            self.state.page_content = self.sites[target_site]['pages']['home']
            return 0.1
        
        return -0.05  # Small penalty for irrelevant site
    
    def _execute_save_info(self, params: Dict) -> float:
        """Save information for later use"""
        if 'key' not in params or 'value' not in params:
            return -0.1
        
        self.state.information_gathered[params['key']] = params['value']
        
        # Reward for saving relevant information
        if params['key'] in ['price', 'product_id', 'flight_id', 'hotel_id']:
            return 0.3
        
        return 0.1
    
    def _execute_compare(self) -> float:
        """Compare saved information"""
        if 'prices' in self.state.information_gathered:
            prices = self.state.information_gathered['prices']
            if isinstance(prices, list) and len(prices) > 1:
                best_price = min(prices)
                self.state.information_gathered['best_price'] = best_price
                return 0.5  # Good reward for comparison
        
        return 0.0
    
    def _execute_submit(self) -> float:
        """Submit form or complete transaction"""
        if self.state.current_page in ['product', 'checkout']:
            # Check if we're making an informed decision
            if 'best_price' in self.state.information_gathered:
                return 2.0  # Large reward for informed purchase
            return 0.5
        
        return 0.1
    
    def _update_task_progress(self) -> None:
        """Update subtask completion based on current state"""
        # Simple heuristic-based progress tracking
        info = self.state.information_gathered
        
        for i, subtask in enumerate(self.current_task.subtasks):
            if not self.state.task_progress[subtask]:
                if 'search' in subtask.lower() and 'search_results' in info:
                    self.state.task_progress[subtask] = True
                elif 'compare' in subtask.lower() and 'best_price' in info:
                    self.state.task_progress[subtask] = True
                elif 'book' in subtask.lower() and self.state.current_page == 'checkout':
                    self.state.task_progress[subtask] = True
    
    def _calculate_progress_reward(self) -> float:
        """Calculate reward based on task progress"""
        completed = sum(self.state.task_progress.values())
        total = len(self.state.task_progress)
        
        # Reward for each completed subtask
        return (completed / total) * 0.5
    
    def get_available_actions(self) -> List[int]:
        """Get valid actions for current state"""
        # Most actions are always available in WebArena
        return list(self.ACTIONS.values())
    
    def render(self, mode: str = 'text') -> str:
        """Render current state"""
        output = [
            "=" * 50,
            f"Task: {self.current_task.name}",
            f"Current: {self.state.to_text()}",
            f"Progress: {sum(self.state.task_progress.values())}/{len(self.state.task_progress)} subtasks",
            f"Info gathered: {list(self.state.information_gathered.keys())}",
            "=" * 50
        ]
        return '\n'.join(output)
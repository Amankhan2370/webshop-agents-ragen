"""
Failure analysis for WebArena tasks
Identifies why RAGEN fails on complex web navigation
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path


class FailureAnalyzer:
    """
    Analyze failure modes in WebArena tasks
    """
    
    def __init__(self):
        """Initialize failure analyzer"""
        self.failure_categories = {
            'planning': 'Long-horizon planning failure',
            'navigation': 'Multi-domain navigation error',
            'information_loss': 'Lost information across sites',
            'action_selection': 'Invalid action for current state',
            'timeout': 'Exceeded maximum steps',
            'credit_assignment': 'Failed to attribute reward to actions',
            'exploration': 'Insufficient exploration of action space'
        }
        
        self.failure_patterns = []
        
    def analyze_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        final_state: Any,
        task: Any
    ) -> Dict[str, Any]:
        """
        Analyze a failed trajectory to identify failure mode
        
        Args:
            trajectory: List of state-action-reward tuples
            final_state: Final state when episode ended
            task: Task that was being attempted
            
        Returns:
            Dictionary with failure analysis
        """
        failure_info = {
            'task_id': task.id if hasattr(task, 'id') else 'unknown',
            'task_complexity': task.complexity if hasattr(task, 'complexity') else 'unknown',
            'trajectory_length': len(trajectory),
            'type': 'unknown',
            'description': '',
            'critical_step': -1,
            'recovery_possible': False
        }
        
        # Check for timeout
        if len(trajectory) >= 30:  # Assuming max_steps = 30
            failure_info['type'] = 'timeout'
            failure_info['description'] = 'Task exceeded maximum allowed steps'
            
        # Check for planning failure
        elif self._detect_planning_failure(trajectory):
            failure_info['type'] = 'planning'
            failure_info['description'] = 'Agent lost track of overall goal'
            failure_info['critical_step'] = self._find_critical_planning_step(trajectory)
            
        # Check for navigation failure
        elif self._detect_navigation_failure(trajectory):
            failure_info['type'] = 'navigation'
            failure_info['description'] = 'Failed to navigate between sites correctly'
            failure_info['critical_step'] = self._find_navigation_error_step(trajectory)
            
        # Check for information loss
        elif self._detect_information_loss(trajectory, final_state):
            failure_info['type'] = 'information_loss'
            failure_info['description'] = 'Lost critical information gathered earlier'
            
        # Check for action selection failure
        elif self._detect_action_failure(trajectory):
            failure_info['type'] = 'action_selection'
            failure_info['description'] = 'Selected invalid or suboptimal actions'
            
        # Analyze if recovery was possible
        failure_info['recovery_possible'] = self._check_recovery_possibility(
            trajectory, 
            failure_info['critical_step']
        )
        
        return failure_info
    
    def _detect_planning_failure(self, trajectory: List[Dict]) -> bool:
        """Detect if failure was due to planning issues"""
        if len(trajectory) < 10:
            return False
        
        # Check for repetitive actions (sign of being stuck)
        recent_actions = [t.get('action', -1) for t in trajectory[-5:]]
        if len(set(recent_actions)) == 1:
            return True
        
        # Check for backtracking without progress
        states = [t.get('state', '') for t in trajectory]
        if len(states) > 10:
            # Count how many times we return to same state
            state_counts = Counter(states)
            if max(state_counts.values()) > 3:
                return True
        
        return False
    
    def _detect_navigation_failure(self, trajectory: List[Dict]) -> bool:
        """Detect if failure was due to navigation between sites"""
        # Look for patterns indicating navigation confusion
        states = [t.get('state', '') for t in trajectory]
        
        # Check if agent keeps switching sites without purpose
        site_switches = 0
        for i in range(1, len(states)):
            if 'switch' in str(trajectory[i].get('action', '')):
                site_switches += 1
        
        # Too many switches indicates navigation confusion
        return site_switches > len(trajectory) * 0.3
    
    def _detect_information_loss(
        self,
        trajectory: List[Dict],
        final_state: Any
    ) -> bool:
        """Detect if critical information was lost"""
        # Check if information was gathered but not used
        gathered_info = False
        used_info = False
        
        for t in trajectory:
            action_str = str(t.get('action', ''))
            if 'save' in action_str or 'copy' in action_str:
                gathered_info = True
            if 'compare' in action_str or 'paste' in action_str:
                used_info = True
        
        # Information was gathered but never used
        return gathered_info and not used_info
    
    def _detect_action_failure(self, trajectory: List[Dict]) -> bool:
        """Detect if failure was due to poor action selection"""
        # Check for invalid actions (negative rewards)
        negative_rewards = sum(1 for t in trajectory if t.get('reward', 0) < 0)
        
        # High proportion of negative rewards indicates action selection issues
        return negative_rewards > len(trajectory) * 0.4
    
    def _find_critical_planning_step(self, trajectory: List[Dict]) -> int:
        """Find the step where planning went wrong"""
        # Look for where agent started repeating actions
        for i in range(len(trajectory) - 5):
            window = trajectory[i:i+5]
            actions = [t.get('action', -1) for t in window]
            if len(set(actions)) <= 2:  # Repetitive actions
                return i
        return -1
    
    def _find_navigation_error_step(self, trajectory: List[Dict]) -> int:
        """Find the step where navigation went wrong"""
        for i, t in enumerate(trajectory):
            if 'switch' in str(t.get('action', '')):
                # Check if this switch led to lower rewards
                if i > 0 and t.get('reward', 0) < trajectory[i-1].get('reward', 0):
                    return i
        return -1
    
    def _check_recovery_possibility(
        self,
        trajectory: List[Dict],
        critical_step: int
    ) -> bool:
        """Check if recovery was possible after critical failure"""
        if critical_step < 0 or critical_step >= len(trajectory) - 5:
            return False
        
        # Check if there were still enough steps to complete task
        remaining_steps = len(trajectory) - critical_step
        return remaining_steps > 5
    
    def find_patterns(self, failures: List[Dict]) -> List[Dict[str, Any]]:
        """Find common patterns across multiple failures"""
        patterns = []
        
        # Group failures by type
        failure_groups = defaultdict(list)
        for f in failures:
            failure_groups[f.get('type', 'unknown')].append(f)
        
        # Analyze each group
        for failure_type, group in failure_groups.items():
            if len(group) < 2:
                continue
                
            pattern = {
                'type': failure_type,
                'frequency': len(group),
                'percentage': (len(group) / len(failures)) * 100,
                'avg_trajectory_length': np.mean([f['trajectory_length'] for f in group]),
                'recovery_rate': np.mean([f['recovery_possible'] for f in group]) * 100
            }
            patterns.append(pattern)
        
        # Sort by frequency
        patterns.sort(key=lambda x: x['frequency'], reverse=True)
        
        return patterns
    
    def generate_failure_report(
        self,
        failures: List[Dict],
        save_path: Optional[str] = None
    ) -> str:
        """Generate comprehensive failure analysis report"""
        report_lines = [
            "=" * 60,
            "Failure Analysis Report",
            "=" * 60,
            f"Total Failures Analyzed: {len(failures)}",
            "",
            "## Failure Type Distribution",
        ]
        
        # Count failure types
        type_counts = Counter(f.get('type', 'unknown') for f in failures)
        total = sum(type_counts.values())
        
        for failure_type, count in type_counts.most_common():
            percentage = (count / total) * 100
            description = self.failure_categories.get(failure_type, 'Unknown failure')
            report_lines.append(f"  {failure_type}: {count} ({percentage:.1f}%) - {description}")
        
        # Analyze patterns
        patterns = self.find_patterns(failures)
        
        report_lines.append("")
        report_lines.append("## Common Patterns")
        
        for i, pattern in enumerate(patterns[:5], 1):
            report_lines.append(f"\n{i}. {pattern['type'].upper()} Pattern")
            report_lines.append(f"   Frequency: {pattern['frequency']} failures")
            report_lines.append(f"   Percentage: {pattern['percentage']:.1f}%")
            report_lines.append(f"   Avg Trajectory Length: {pattern['avg_trajectory_length']:.1f}")
            report_lines.append(f"   Recovery Rate: {pattern['recovery_rate']:.1f}%")
        
        # Complexity analysis
        report_lines.append("")
        report_lines.append("## Failures by Task Complexity")
        
        complexity_groups = defaultdict(list)
        for f in failures:
            complexity_groups[f.get('task_complexity', 'unknown')].append(f)
        
        for complexity in ['easy', 'medium', 'hard']:
            if complexity in complexity_groups:
                group = complexity_groups[complexity]
                report_lines.append(f"  {complexity.capitalize()}: {len(group)} failures")
                
                # Most common failure type for this complexity
                types = [f.get('type', 'unknown') for f in group]
                if types:
                    most_common = Counter(types).most_common(1)[0]
                    report_lines.append(f"    Most common: {most_common[0]} ({most_common[1]} times)")
        
        # Recovery analysis
        report_lines.append("")
        report_lines.append("## Recovery Analysis")
        
        recoverable = sum(1 for f in failures if f.get('recovery_possible', False))
        recovery_rate = (recoverable / len(failures)) * 100 if failures else 0
        
        report_lines.append(f"  Recoverable Failures: {recoverable}/{len(failures)} ({recovery_rate:.1f}%)")
        
        # Critical steps analysis
        critical_steps = [f.get('critical_step', -1) for f in failures if f.get('critical_step', -1) > 0]
        if critical_steps:
            report_lines.append(f"  Average Critical Step: {np.mean(critical_steps):.1f}")
            report_lines.append(f"  Earliest Critical Step: {min(critical_steps)}")
            report_lines.append(f"  Latest Critical Step: {max(critical_steps)}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def suggest_improvements(self, failures: List[Dict]) -> List[str]:
        """Suggest improvements based on failure analysis"""
        suggestions = []
        
        # Analyze failure distribution
        type_counts = Counter(f.get('type', 'unknown') for f in failures)
        total = sum(type_counts.values())
        
        for failure_type, count in type_counts.most_common():
            percentage = (count / total) * 100
            
            if failure_type == 'planning' and percentage > 30:
                suggestions.append(
                    "Implement hierarchical planning to handle long-horizon tasks"
                )
                suggestions.append(
                    "Add task decomposition to break complex goals into subtasks"
                )
                
            elif failure_type == 'navigation' and percentage > 25:
                suggestions.append(
                    "Improve state representation to maintain context across sites"
                )
                suggestions.append(
                    "Add site-specific adapters for better cross-domain transfer"
                )
                
            elif failure_type == 'information_loss' and percentage > 20:
                suggestions.append(
                    "Implement external memory module to store cross-site information"
                )
                suggestions.append(
                    "Add attention mechanism over historical information"
                )
                
            elif failure_type == 'action_selection' and percentage > 30:
                suggestions.append(
                    "Increase exploration during training"
                )
                suggestions.append(
                    "Use curriculum learning to gradually increase task complexity"
                )
                
            elif failure_type == 'timeout' and percentage > 25:
                suggestions.append(
                    "Optimize action selection policy for efficiency"
                )
                suggestions.append(
                    "Add time-aware rewards to encourage faster completion"
                )
        
        # General suggestions
        if len(failures) > 50:
            suggestions.append(
                "Consider pre-training on simpler web navigation tasks"
            )
            suggestions.append(
                "Implement reward shaping for intermediate subtask completion"
            )
        
        return suggestions
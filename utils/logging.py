"""
Logging utilities for training and evaluation
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
import wandb


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        format_str: Custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create format
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_str)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")
    
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: int,
    prefix: str = ""
):
    """
    Log metrics to logger
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        step: Current step/episode
        prefix: Prefix for metric names
    """
    metric_str = f"Step {step:6d} | "
    
    for key, value in metrics.items():
        if isinstance(value, float):
            metric_str += f"{prefix}{key}: {value:.4f} | "
        else:
            metric_str += f"{prefix}{key}: {value} | "
    
    logger.info(metric_str)


class WandbLogger:
    """
    Weights & Biases logger for experiment tracking
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
        mode: str = "online"
    ):
        """
        Initialize W&B logger
        
        Args:
            project: W&B project name
            name: Run name
            config: Configuration dictionary
            entity: W&B entity
            tags: List of tags
            notes: Run notes
            mode: "online", "offline", or "disabled"
        """
        self.enabled = mode != "disabled"
        
        if self.enabled:
            try:
                wandb.init(
                    project=project,
                    name=name,
                    config=config,
                    entity=entity,
                    tags=tags,
                    notes=notes,
                    mode=mode
                )
                self.run = wandb.run
                print(f"W&B run initialized: {self.run.url}")
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")
                self.enabled = False
                self.run = None
        else:
            self.run = None
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B"""
        if self.enabled and self.run:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def log_table(self, table_name: str, data: list):
        """Log table data to W&B"""
        if self.enabled and self.run:
            table = wandb.Table(data=data)
            wandb.log({table_name: table})
    
    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log artifact to W&B"""
        if self.enabled and self.run:
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=metadata
            )
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish W&B run"""
        if self.enabled and self.run:
            wandb.finish()


class MetricsLogger:
    """
    Combined metrics logger for file and W&B logging
    """
    
    def __init__(
        self,
        log_dir: str,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory for log files
            use_wandb: Whether to use W&B
            wandb_config: W&B configuration
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.logger = setup_logger("metrics", log_dir)
        
        # Setup metrics file
        self.metrics_file = self.log_dir / "metrics.jsonl"
        
        # Setup W&B if requested
        self.use_wandb = use_wandb
        if use_wandb:
            wandb_config = wandb_config or {}
            self.wandb_logger = WandbLogger(**wandb_config)
        else:
            self.wandb_logger = None
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: int,
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_to_wandb: bool = True
    ):
        """
        Log metrics to multiple destinations
        
        Args:
            metrics: Metrics dictionary
            step: Current step
            log_to_console: Log to console
            log_to_file: Log to file
            log_to_wandb: Log to W&B
        """
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}
        
        # Console logging
        if log_to_console:
            log_metrics(self.logger, metrics, step)
        
        # File logging
        if log_to_file:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics_with_step) + '\n')
        
        # W&B logging
        if log_to_wandb and self.wandb_logger:
            self.wandb_logger.log(metrics, step)
    
    def log_summary(self, summary: Dict[str, Any]):
        """Log summary statistics"""
        self.logger.info("=" * 60)
        self.logger.info("Summary Statistics")
        self.logger.info("=" * 60)
        
        for key, value in summary.items():
            self.logger.info(f"{key}: {value}")
        
        # Save summary to file
        summary_file = self.log_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def finish(self):
        """Finish logging"""
        if self.wandb_logger:
            self.wandb_logger.finish()


def create_experiment_name(
    base_name: str,
    config: Dict[str, Any]
) -> str:
    """
    Create experiment name from base name and config
    
    Args:
        base_name: Base experiment name
        config: Configuration dictionary
        
    Returns:
        Experiment name with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add key config values to name
    key_params = []
    
    if 'model' in config:
        if 'hidden_dim' in config['model']:
            key_params.append(f"h{config['model']['hidden_dim']}")
        if 'num_layers' in config['model']:
            key_params.append(f"l{config['model']['num_layers']}")
    
    if 'training' in config:
        if 'learning_rate' in config['training']:
            lr = config['training']['learning_rate']
            key_params.append(f"lr{lr:.0e}")
    
    if key_params:
        param_str = "_".join(key_params)
        return f"{base_name}_{param_str}_{timestamp}"
    else:
        return f"{base_name}_{timestamp}"


def log_hyperparameters(
    logger: logging.Logger,
    config: Dict[str, Any]
):
    """
    Log hyperparameters
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("Hyperparameters")
    logger.info("=" * 60)
    
    def log_dict(d: dict, prefix: str = ""):
        for key, value in d.items():
            if isinstance(value, dict):
                log_dict(value, f"{prefix}{key}.")
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 60)


def setup_tensorboard(log_dir: str):
    """
    Setup TensorBoard logging
    
    Args:
        log_dir: Directory for TensorBoard logs
        
    Returns:
        SummaryWriter instance
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        tb_dir = Path(log_dir) / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        
        writer = SummaryWriter(tb_dir)
        print(f"TensorBoard logging to {tb_dir}")
        print(f"Run: tensorboard --logdir {tb_dir}")
        
        return writer
    except ImportError:
        print("TensorBoard not available")
        return None


class ProgressLogger:
    """
    Logger for training progress with ETA estimation
    """
    
    def __init__(
        self,
        total_steps: int,
        log_interval: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize progress logger
        
        Args:
            total_steps: Total number of steps
            log_interval: Logging interval
            logger: Logger instance
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.logger = logger or logging.getLogger(__name__)
        
        self.start_time = datetime.now()
        self.step_times = []
        
    def log(self, step: int, metrics: Dict[str, Any]):
        """Log progress with ETA"""
        if step % self.log_interval != 0 and step != self.total_steps:
            return
        
        # Calculate timing
        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_sec = step / elapsed if elapsed > 0 else 0
        
        if steps_per_sec > 0:
            remaining_steps = self.total_steps - step
            eta_seconds = remaining_steps / steps_per_sec
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "Unknown"
        
        # Format metrics
        metric_str = " | ".join([
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        ])
        
        # Log progress
        progress = (step / self.total_steps) * 100
        self.logger.info(
            f"Step {step}/{self.total_steps} ({progress:.1f}%) | "
            f"ETA: {eta_str} | {metric_str}"
        )
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to readable time"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
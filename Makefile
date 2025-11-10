# Makefile for WebShop-WebArena RAGEN Project

.PHONY: help install train evaluate test clean lint format all

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := webshop-webarena-ragen
CONFIG := training/config.yaml
OUTPUT_DIR := experiments/results

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help:
	@echo "$(GREEN)WebShop-WebArena RAGEN Makefile$(NC)"
	@echo "================================"
	@echo "Available commands:"
	@echo "  $(YELLOW)make install$(NC)    - Install dependencies"
	@echo "  $(YELLOW)make train$(NC)      - Train RAGEN on WebShop"
	@echo "  $(YELLOW)make evaluate$(NC)   - Evaluate on WebArena"
	@echo "  $(YELLOW)make test$(NC)       - Run unit tests"
	@echo "  $(YELLOW)make demo$(NC)       - Run interactive demo"
	@echo "  $(YELLOW)make clean$(NC)      - Clean generated files"
	@echo "  $(YELLOW)make lint$(NC)       - Check code style"
	@echo "  $(YELLOW)make format$(NC)     - Format code"
	@echo "  $(YELLOW)make all$(NC)        - Run all experiments"

install:
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "$(GREEN)Installation complete!$(NC)"

train:
	@echo "$(GREEN)Training RAGEN on WebShop...$(NC)"
	$(PYTHON) training/train_webshop.py \
		--config $(CONFIG) \
		--output_dir $(OUTPUT_DIR)/webshop
	@echo "$(GREEN)Training complete!$(NC)"

evaluate:
	@echo "$(GREEN)Evaluating on WebArena...$(NC)"
	$(PYTHON) evaluation/evaluate_webarena.py \
		--config $(CONFIG) \
		--checkpoint $(OUTPUT_DIR)/webshop/best_model.pt \
		--output_dir $(OUTPUT_DIR)/webarena
	@echo "$(GREEN)Evaluation complete!$(NC)"

test:
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=./ --cov-report=html
	@echo "$(GREEN)Tests complete! Coverage report in htmlcov/index.html$(NC)"

demo:
	@echo "$(GREEN)Starting interactive demo...$(NC)"
	$(PYTHON) main.py demo --config $(CONFIG)

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf logs/* checkpoints/* experiments/results/*
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*~" -delete
	@echo "$(GREEN)Cleanup complete!$(NC)"

lint:
	@echo "$(GREEN)Checking code style...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	black --line-length 100 .
	isort . --profile black --line-length 100

# Run all experiments
all:
	@echo "$(GREEN)Running all experiments...$(NC)"
	$(PYTHON) experiments/run_all.py \
		--config $(CONFIG) \
		--output_dir $(OUTPUT_DIR)/all
	@echo "$(GREEN)All experiments complete!$(NC)"

# Quick test run with debug mode
debug:
	@echo "$(YELLOW)Running in debug mode...$(NC)"
	$(PYTHON) main.py train --config $(CONFIG) --debug

# Generate performance plots
plots:
	@echo "$(GREEN)Generating performance plots...$(NC)"
	$(PYTHON) -c "from evaluation.metrics import MetricsTracker; \
		tracker = MetricsTracker('$(OUTPUT_DIR)'); \
		tracker.load(); \
		tracker.plot_training_curves(); \
		tracker.plot_evaluation_results()"
	@echo "$(GREEN)Plots saved to $(OUTPUT_DIR)$(NC)"

# Download pretrained models (if available)
download-models:
	@echo "$(GREEN)Downloading pretrained models...$(NC)"
	mkdir -p checkpoints/pretrained
	# Add wget/curl commands here to download models
	@echo "$(YELLOW)Note: Add model URLs to Makefile$(NC)"

# Create submission package
submission:
	@echo "$(GREEN)Creating submission package...$(NC)"
	mkdir -p submission
	cp -r models algorithms agents environments submission/
	cp -r training evaluation submission/
	cp README.md requirements.txt setup.py submission/
	tar -czf submission.tar.gz submission/
	@echo "$(GREEN)Submission package created: submission.tar.gz$(NC)"

# Check GPU availability
check-gpu:
	@echo "$(GREEN)Checking GPU availability...$(NC)"
	@$(PYTHON) -c "import torch; \
		print('CUDA available:', torch.cuda.is_available()); \
		print('GPU count:', torch.cuda.device_count()); \
		print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Initialize project structure
init:
	@echo "$(GREEN)Initializing project structure...$(NC)"
	mkdir -p data/{webshop,webarena}
	mkdir -p logs checkpoints experiments/results
	mkdir -p tests notebooks presentation
	touch logs/.gitkeep checkpoints/.gitkeep
	@echo "$(GREEN)Project structure initialized!$(NC)"
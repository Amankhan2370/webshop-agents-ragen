#!/bin/bash

# WebShop-WebArena RAGEN Training and Evaluation Script
# This script runs the complete pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="training/config.yaml"
OUTPUT_DIR="experiments/results"
LOG_DIR="logs"

# Print banner
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         WebShop-WebArena RAGEN Pipeline                  ║"
echo "║     Reasoning via A*-guided Planning with A*PO          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print section headers
print_header() {
    echo -e "\n${YELLOW}==== $1 ====${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_header "Checking Prerequisites"

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

if ! command_exists pip3; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"

# Parse command line arguments
COMMAND=${1:-"all"}
DEBUG=${2:-""}

case $COMMAND in
    install)
        print_header "Installing Dependencies"
        pip3 install -r requirements.txt
        echo -e "${GREEN}✓ Dependencies installed${NC}"
        ;;
        
    train)
        print_header "Training RAGEN on WebShop"
        
        # Create output directory
        mkdir -p ${OUTPUT_DIR}/webshop
        mkdir -p ${LOG_DIR}
        
        # Run training
        if [ "$DEBUG" == "--debug" ]; then
            echo "Running in debug mode (fewer episodes)..."
            python3 training/train_webshop.py \
                --config ${CONFIG_FILE} \
                --output_dir ${OUTPUT_DIR}/webshop \
                --debug 2>&1 | tee ${LOG_DIR}/train_webshop.log
        else
            python3 training/train_webshop.py \
                --config ${CONFIG_FILE} \
                --output_dir ${OUTPUT_DIR}/webshop \
                2>&1 | tee ${LOG_DIR}/train_webshop.log
        fi
        
        echo -e "${GREEN}✓ Training complete${NC}"
        echo "Log saved to: ${LOG_DIR}/train_webshop.log"
        ;;
        
    evaluate)
        print_header "Evaluating on WebArena"
        
        # Find checkpoint
        CHECKPOINT="${OUTPUT_DIR}/webshop/best_model.pt"
        if [ ! -f "$CHECKPOINT" ]; then
            CHECKPOINT="${OUTPUT_DIR}/webshop/final_model.pt"
        fi
        
        if [ ! -f "$CHECKPOINT" ]; then
            echo -e "${RED}Error: No checkpoint found. Please train first.${NC}"
            exit 1
        fi
        
        # Create output directory
        mkdir -p ${OUTPUT_DIR}/webarena
        
        # Run evaluation
        python3 evaluation/evaluate_webarena.py \
            --config ${CONFIG_FILE} \
            --checkpoint ${CHECKPOINT} \
            --output_dir ${OUTPUT_DIR}/webarena \
            --num_episodes 10 \
            2>&1 | tee ${LOG_DIR}/evaluate_webarena.log
        
        echo -e "${GREEN}✓ Evaluation complete${NC}"
        echo "Log saved to: ${LOG_DIR}/evaluate_webarena.log"
        ;;
        
    baseline)
        print_header "Running Baseline Comparisons"
        
        # Run baseline agents
        for AGENT in random rule_based heuristic; do
            echo -e "\n${YELLOW}Evaluating ${AGENT} baseline...${NC}"
            
            python3 evaluation/evaluate_webshop.py \
                --agent_type ${AGENT} \
                --output_dir ${OUTPUT_DIR}/baseline_${AGENT} \
                --num_episodes 50 \
                2>&1 | tee ${LOG_DIR}/baseline_${AGENT}.log
        done
        
        echo -e "${GREEN}✓ Baseline comparisons complete${NC}"
        ;;
        
    all)
        print_header "Running Complete Pipeline"
        
        # Run all steps
        echo "Step 1/4: Training on WebShop..."
        $0 train $DEBUG
        
        echo -e "\nStep 2/4: Evaluating on WebArena..."
        $0 evaluate
        
        echo -e "\nStep 3/4: Running baselines..."
        $0 baseline
        
        echo -e "\nStep 4/4: Generating report..."
        $0 report
        
        echo -e "${GREEN}✓ Pipeline complete!${NC}"
        ;;
        
    report)
        print_header "Generating Final Report"
        
        # Run report generation
        python3 experiments/run_all.py \
            --config ${CONFIG_FILE} \
            --output_dir ${OUTPUT_DIR}/final \
            --skip_training \
            2>&1 | tee ${LOG_DIR}/report.log
        
        echo -e "${GREEN}✓ Report generated${NC}"
        echo "Report saved to: ${OUTPUT_DIR}/final/experiment_report.txt"
        ;;
        
    demo)
        print_header "Running Interactive Demo"
        
        # Find checkpoint
        CHECKPOINT="${OUTPUT_DIR}/webshop/best_model.pt"
        
        python3 main.py demo \
            --config ${CONFIG_FILE} \
            --checkpoint ${CHECKPOINT}
        ;;
        
    clean)
        print_header "Cleaning Generated Files"
        
        echo "Removing logs..."
        rm -rf ${LOG_DIR}/*
        
        echo "Removing checkpoints..."
        rm -rf checkpoints/*
        
        echo "Removing results..."
        rm -rf ${OUTPUT_DIR}/*
        
        echo "Removing Python cache..."
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete
        
        echo -e "${GREEN}✓ Cleanup complete${NC}"
        ;;
        
    test)
        print_header "Running Tests"
        
        # Run pytest
        python3 -m pytest tests/ -v --cov=./ --cov-report=term-missing
        
        echo -e "${GREEN}✓ Tests complete${NC}"
        ;;
        
    *)
        echo "Usage: $0 {install|train|evaluate|baseline|all|report|demo|clean|test} [--debug]"
        echo ""
        echo "Commands:"
        echo "  install   - Install dependencies"
        echo "  train     - Train RAGEN on WebShop"
        echo "  evaluate  - Evaluate on WebArena"
        echo "  baseline  - Run baseline comparisons"
        echo "  all       - Run complete pipeline"
        echo "  report    - Generate final report"
        echo "  demo      - Run interactive demo"
        echo "  clean     - Clean generated files"
        echo "  test      - Run unit tests"
        echo ""
        echo "Options:"
        echo "  --debug   - Run in debug mode (fewer episodes)"
        exit 1
        ;;
esac

# Print summary
echo -e "\n${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}    Execution completed successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}\n"
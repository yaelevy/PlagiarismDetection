#!/bin/bash

# Plagiarism Detection Inference Script
# Usage: ./run_inference.sh [options]

# Default values
MODEL_PATH="./best_siamese_bert.pth"
CANDIDATES_FILE="bloom_candidates.json"
THRESHOLD=0.5
MAX_CANDIDATES=""
OUTPUT_FILE=""
TEST_SINGLE=""

# Help function
show_help() {
    echo "Plagiarism Detection Inference Pipeline"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model_path PATH         Path to trained model (default: $MODEL_PATH)"
    echo "  -c, --candidates_file FILE    Path to candidates JSON (default: $CANDIDATES_FILE)"
    echo "  -t, --threshold FLOAT         Similarity threshold (default: $THRESHOLD)"
    echo "  -n, --max_candidates N        Max candidates to process (default: all)"
    echo "  -o, --output_file FILE        Output file path (default: auto-generated)"
    echo "  --test PAPER_A PAPER_B        Test single pair of papers"
    echo "  --quick N                     Quick test with top N candidates (default: 10)"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Run inference on all candidates"
    echo "  $0"
    echo ""
    echo "  # Run inference on top 50 candidates"
    echo "  $0 --max_candidates 50"
    echo ""
    echo "  # Quick test with top 10 candidates"
    echo "  $0 --quick 10"
    echo ""
    echo "  # Test specific pair"
    echo "  $0 --test paper1.tex paper2.tex"
    echo ""
    echo "  # Use custom model and threshold"
    echo "  $0 -m ./my_model.pth -t 0.7"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--candidates_file)
            CANDIDATES_FILE="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -n|--max_candidates)
            MAX_CANDIDATES="$2"
            shift 2
            ;;
        -o|--output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --test)
            TEST_PAPER_A="$2"
            TEST_PAPER_B="$3"
            shift 3
            ;;
        --quick)
            MAX_CANDIDATES="${2:-10}"
            echo "Quick test mode: processing top $MAX_CANDIDATES candidates"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if Python script exists
PYTHON_SCRIPT="inference_pipeline.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found in current directory"
    echo "Please ensure the Python script is in the same directory as this wrapper"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found"
    echo "Please train the model first or specify correct path with -m/--model_path"
    exit 1
fi

# Check dependencies
echo "Checking Python dependencies..."
python3 -c "import torch, transformers, pandas, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    pip install torch transformers pandas tqdm
fi

# Build command
CMD="python3 $PYTHON_SCRIPT"
CMD="$CMD --model_path \"$MODEL_PATH\""
CMD="$CMD --threshold $THRESHOLD"

# Add test mode if specified
if [ ! -z "$TEST_PAPER_A" ] && [ ! -z "$TEST_PAPER_B" ]; then
    CMD="$CMD --test_single \"$TEST_PAPER_A\" \"$TEST_PAPER_B\""
    echo "==================================="
    echo "Single Pair Testing Mode"
    echo "==================================="
    echo "Paper A: $TEST_PAPER_A"
    echo "Paper B: $TEST_PAPER_B"
    echo "Model: $MODEL_PATH"
    echo "Threshold: $THRESHOLD"
    echo "==================================="
    echo ""
else
    # Regular inference mode
    if [ ! -f "$CANDIDATES_FILE" ]; then
        echo "Error: Candidates file $CANDIDATES_FILE not found"
        echo "Please run Bloom filter preprocessing first"
        exit 1
    fi
    
    CMD="$CMD --candidates_file \"$CANDIDATES_FILE\""
    
    if [ ! -z "$MAX_CANDIDATES" ]; then
        CMD="$CMD --max_candidates $MAX_CANDIDATES"
    fi
    
    if [ ! -z "$OUTPUT_FILE" ]; then
        CMD="$CMD --output_file \"$OUTPUT_FILE\""
    fi
    
    echo "==================================="
    echo "Plagiarism Detection Inference"
    echo "==================================="
    echo "Model: $MODEL_PATH"
    echo "Candidates: $CANDIDATES_FILE"
    echo "Threshold: $THRESHOLD"
    if [ ! -z "$MAX_CANDIDATES" ]; then
        echo "Max Candidates: $MAX_CANDIDATES"
    else
        echo "Max Candidates: All"
    fi
    if [ ! -z "$OUTPUT_FILE" ]; then
        echo "Output: $OUTPUT_FILE"
    else
        echo "Output: Auto-generated"
    fi
    echo "==================================="
    echo ""
    
    # Count candidates
    CANDIDATE_COUNT=$(python3 -c "import json; print(len(json.load(open('$CANDIDATES_FILE'))))" 2>/dev/null)
    if [ ! -z "$CANDIDATE_COUNT" ]; then
        if [ ! -z "$MAX_CANDIDATES" ]; then
            echo "Processing top $MAX_CANDIDATES of $CANDIDATE_COUNT total candidates"
        else
            echo "Processing all $CANDIDATE_COUNT candidates"
        fi
        echo ""
    fi
fi

# Run the command
echo "Starting inference..."
echo "Command: $CMD"
echo ""

eval $CMD

# Check if inference completed successfully
if [ $? -eq 0 ]; then
    if [ -z "$TEST_PAPER_A" ]; then
        echo ""
        echo "==================================="
        echo "Inference completed successfully!"
        echo "==================================="
        echo "Check the output files for detailed results."
        echo ""
        echo "Quick analysis tips:"
        echo "  • High similarity scores (>0.7) are strong plagiarism indicators"
        echo "  • Compare BERT scores with Bloom filter signals"
        echo "  • Investigate high-scoring pairs with different authors"
        echo ""
    fi
else
    echo ""
    echo "Error: Inference failed!"
    exit 1
fi

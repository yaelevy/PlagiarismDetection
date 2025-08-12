#!/bin/bash

# Plagiarism Detection Training Script
# Usage: ./run_plagiarism_detection.sh [options]

# Default values
CORPUS_PATH="/Users/yaronot/Desktop/Data Courses/Plagiarism Detection/pan-plagiarism-corpus-2011-1/external-detection-corpus"
CSV_FILE="all_plagiarism_mappings.csv"
OUTPUT_DIR="./plagiarism_output"
SAMPLE_SIZE=20000
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=2e-5
THRESHOLD=0.5
TRAIN_JSON=""
VAL_JSON=""



# Help function
show_help() {
    echo "Plagiarism Detection with Siamese BERT"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --corpus_path PATH    Path to corpus directory (default: $CORPUS_PATH)"
    echo "  -f, --csv_file FILE       Path to plagiarism mappings CSV (default: $CSV_FILE)"
    echo "  -o, --output_dir DIR      Output directory (default: $OUTPUT_DIR)"
    echo "  -s, --sample_size N       Sample size for training (default: $SAMPLE_SIZE)"
    echo "  -e, --epochs N            Number of epochs (default: $EPOCHS)"
    echo "  -b, --batch_size N        Batch size (default: $BATCH_SIZE)"
    echo "  -l, --learning_rate F     Learning rate (default: $LEARNING_RATE)"
    echo "  --test TEXT1 TEXT2        Test similarity between two texts"
    echo "  --load_model PATH         Load pre-trained model for inference"
    echo "  -h, --help               Show this help message"
    echo "  --train_json PATH         Path to pre-generated training pairs JSON"
    echo "  --val_json PATH           Path to pre-generated validation pairs JSON"
    echo ""
    echo "Examples:"
    echo "  # Train with default settings"
    echo "  $0"
    echo ""
    echo "  # Train with custom corpus path and sample size"
    echo "  $0 -c /path/to/corpus -s 2000"
    echo ""
    echo "  # Test similarity with pre-trained model"
    echo "  $0 --load_model ./best_siamese_bert.pth --test \"Text 1\" \"Text 2\""
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--corpus_path)
            CORPUS_PATH="$2"
            shift 2
            ;;
        -f|--csv_file)
            CSV_FILE="$2"
            shift 2
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--sample_size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --test)
            TEST_TEXT1="$2"
            TEST_TEXT2="$3"
            shift 3
            ;;
        --train_json)
            TRAIN_JSON="$2"
            shift 2
            ;;
        --val_json)
            VAL_JSON="$2"
            shift 2
            ;;
        --load_model)
            LOAD_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
	--threshold)
    	    THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if Python script exists
PYTHON_SCRIPT="plagiarism_detector.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found in current directory"
    echo "Please ensure the Python script is in the same directory as this wrapper"
    exit 1
fi

# Check dependencies
echo "Checking Python dependencies..."
python3 -c "import torch, transformers, sklearn, pandas, numpy, matplotlib, seaborn, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Missing dependencies. Installing..."
    pip install torch transformers scikit-learn pandas numpy matplotlib seaborn tqdm
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python3 $PYTHON_SCRIPT"
CMD="$CMD --corpus_path \"$CORPUS_PATH\""
CMD="$CMD --csv_file \"$CSV_FILE\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --sample_size $SAMPLE_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --learning_rate $LEARNING_RATE"
CMD="$CMD --threshold $THRESHOLD"



if [ ! -z "$TRAIN_JSON" ]; then
    CMD="$CMD --train_json \"$TRAIN_JSON\""
fi

if [ ! -z "$VAL_JSON" ]; then
    CMD="$CMD --val_json \"$VAL_JSON\""
fi


# Add inference options if provided
if [ ! -z "$LOAD_MODEL" ]; then
    CMD="$CMD --load_model \"$LOAD_MODEL\""
fi

if [ ! -z "$TEST_TEXT1" ] && [ ! -z "$TEST_TEXT2" ]; then
    CMD="$CMD --test_text1 \"$TEST_TEXT1\" --test_text2 \"$TEST_TEXT2\""
fi

# Check if this is inference mode (has both load_model and test texts)
INFERENCE_MODE=false
if [ ! -z "$LOAD_MODEL" ] && [ ! -z "$TEST_TEXT1" ] && [ ! -z "$TEST_TEXT2" ]; then
    INFERENCE_MODE=true
fi

# Only show configuration for corpus-based training mode
if [ "$INFERENCE_MODE" = false ] && [ -z "$TRAIN_JSON" ] && [ -z "$VAL_JSON" ]; then
    # Show configuration and validate corpus paths
    echo "==================================="
    echo "Plagiarism Detection Configuration"
    echo "==================================="
    echo "Corpus Path: $CORPUS_PATH"
    echo "CSV File: $CSV_FILE"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Sample Size: $SAMPLE_SIZE"
    echo "Epochs: $EPOCHS"
    echo "Batch Size: $BATCH_SIZE"
    echo "Learning Rate: $LEARNING_RATE"
    echo "Threshold: $THRESHOLD"
    echo "==================================="
    echo ""

    # Check if corpus path exists
    if [ ! -d "$CORPUS_PATH" ]; then
        echo "Error: Corpus path $CORPUS_PATH does not exist"
        exit 1
    fi

    # Check if CSV file exists
    if [ ! -f "$CSV_FILE" ]; then
        echo "Error: CSV file $CSV_FILE does not exist"
        exit 1
    fi

    echo "Starting plagiarism detection..."
    echo "Command: $CMD"
    echo ""
else
    echo "Testing similarity..."
fi

eval $CMD

# Check if training completed successfully
if [ $? -eq 0 ]; then
    if [ "$INFERENCE_MODE" = false ]; then
        echo ""
        echo "==================================="
        echo "Training completed successfully!"
        echo "==================================="
        echo "Output files:"
        echo "  - Model: $OUTPUT_DIR/best_siamese_bert.pth"
        echo "  - Training plot: $OUTPUT_DIR/training_history.png"
        echo ""
        echo "To test similarity between two texts:"
        echo "  $0 --load_model $OUTPUT_DIR/best_siamese_bert.pth --test \"Text 1\" \"Text 2\""
    fi
else
    echo ""
    echo "Error: Training failed!"
    exit 1
fi


#!/bin/bash
echo "=== Starting GPU Plagiarism Detection Session ==="

# Request GPU node
echo "Requesting GPU node (this may take a few minutes)..."
srun --partition=catfish --gres=gpu:l4:1 --cpus-per-task=8 --mem=16G --time=3:00:00 --pty bash -c "

# Load environment
echo 'Setting up GPU environment...'
source ~/setup_gpu_plagiarism.sh

# Copy code to work directory
echo 'Copying code files...'
cp ~/apply_plagiarism_detection.py /tmp/plagiarism_work/
cp ~/run_plagiarism_analysis.sh /tmp/plagiarism_work/
cp ~/extract_paragraphs.py /tmp/plagiarism_work/
cp ~/best_siamese_bert.pth /tmp/plagiarism_work/
chmod +x /tmp/plagiarism_work/run_plagiarism_analysis.sh

echo '=== Ready! You are now on GPU node with everything set up ==='
echo 'Run: ./run_plagiarism_analysis.sh --max_articles 10'
echo 'REMEMBER: Copy results back with: cp plagiarism_results.* ~/'

# Start interactive shell
bash
"

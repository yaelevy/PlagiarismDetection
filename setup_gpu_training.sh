#!/bin/bash
echo "=== GPU Training Environment Setup ==="

# Create virtual environment in /tmp (more space)
if [ ! -d "/tmp/plagiarism_train_env" ]; then
    echo "Creating new training environment..."
    python3 -m venv /tmp/plagiarism_train_env
    
    source /tmp/plagiarism_train_env/bin/activate
    module load cuda/12.4.1
    
    echo "Installing CUDA PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers pandas numpy scikit-learn matplotlib seaborn tqdm
else
    echo "Using existing training environment..."
    source /tmp/plagiarism_train_env/bin/activate
    module load cuda/12.4.1
fi

# Set cache directories to /tmp (more space)
export HF_HOME=/tmp/hf_cache_$USER
export TRANSFORMERS_CACHE=/tmp/hf_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER

mkdir -p /tmp/hf_cache_$USER /tmp/torch_cache_$USER /tmp/training_work
cd /tmp/training_work

echo "=== Training setup complete! ==="

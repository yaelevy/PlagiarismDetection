#!/bin/bash
echo "=== Starting GPU Training Session ==="

srun --partition=catfish --gres=gpu:l4:1 --cpus-per-task=8 --mem=16G --time=3:00:00 --pty bash -c "

echo 'Setting up training environment...'
source ~/setup_gpu_training.sh

echo 'Copying training files...'
cp ~/plagiarism_detector.py /tmp/training_work/
cp ~/run_plagiarism_detection.sh /tmp/training_work/
cp ~/train_pairs.json /tmp/training_work/
cp ~/val_pairs.json /tmp/training_work/
chmod +x /tmp/training_work/run_plagiarism_detection.sh

echo '=== Ready for training! ==='
echo 'Run: ./run_plagiarism_detection.sh --train_json train_pairs.json --val_json val_pairs.json --epochs 5 --output_dir /tmp/training_work'
echo 'REMEMBER: Copy model back: cp best_siamese_bert.pth ~/'

bash
"

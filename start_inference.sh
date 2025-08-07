#!/bin/bash
echo "=== Starting Clustering Session ==="

srun --nodelist=goldfish-01 --gres=gpu:h200:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash -c "

echo 'Setting up clustering environment in /tmp...'
cd /tmp
mkdir -p clustering_work
cd clustering_work

echo 'Creating virtual environment...'
python3 -m venv clustering_env
source clustering_env/bin/activate

echo 'Installing packages...'
pip install sentence-transformers umap-learn hdbscan nltk pandas matplotlib seaborn

echo 'Copying clustering script...'
cp ~/tex_clustering.py /tmp/clustering_work/
cp ~/bloom_pipeline.py /tmp/clustering_work/
cp ~/paper_counter.py /tmp/clustering_work/
cp ~/best_siamese_bert.pth /tmp/clustering_work/
cp ~/plagiarism_detector.py /tmp/clustering_work/

echo '=== Ready for clustering! ==='
echo 'Run: python tex_clustering.py /sci/labs/orzuk/orzuk/teaching/big_data_project_52017/2024_25/arxiv_data/full_papers'
echo 'REMEMBER: Copy results back: cp cluster_results.json ~/'

bash
"

# 🧠 Plagiarism Detection Pipeline

This repository contains a complete pipeline for detecting potential plagiarism in LaTeX academic papers using Bloom Filters, Siamese BERT, and clustering techniques.

## 🚀 Training

We moved the training phase to a compute cluster by:
- Predefining a corpus dataset on a local machine
- Uploading it to the cluster
- Creating a dedicated GPU-enabled training environment

## 📁 Files Required in Your Home Folder on the Cluster

Ensure the following files exist in your **home folder** (`~/`) on the cluster:

1. `start_training.sh` — for environment setup and GPU access
2. `setup_gpu_training.sh`
3. `plagiarism_detector.py`
4. `run_plagiarism_detection.sh`
5. `train_pairs.json` — generated locally by running `run_plagiarism_detection.sh` (only the corpus creation stage)
6. `val_pairs.json` — same as above

💡 **Note:** You don't need to train locally — stop the script once the corpus is created.

## 🏋️ Start Training

From the **login node**:

```bash
bash ~/start_training.sh
```

Then on the GPU node:

```bash
bash ./run_plagiarism_detection.sh --train_json train_pairs.json --val_json val_pairs.json --epochs 5 --output_dir /tmp/training_work
```

## 📦 Save Training Outputs

After training (since `/tmp` is volatile), copy the outputs:

```bash
cp /tmp/training_work/best_siamese_bert.pth ~/
cp /tmp/training_work/training_history.png ~/
```

Then exit the training node:

```bash
exit
```

## 🔍 Inference

From the login node:

```bash
bash ~/start_inference.sh
```

This copies necessary scripts and models into `/tmp` for fast access.

### 🧾 Inference Files

- `paper_counter.py` — counts available papers
- `tex_clustering.py` — clusters papers using HDBSCAN
- `bloom_pipeline.py` — Bloom filter preprocessing
- `inference_pipeline.py` — runs inference with Siamese BERT
- `run_plagiarism_weights.sh` — script to apply the trained weights
- `best_siamese_bert.pth` — trained model weights

## 📊 Analysis Steps

### Step 1: Count Available Papers

```bash
python paper_counter.py
```

### Step 2: Clustering (Using HDBSCAN)

```bash
python tex_clustering.py
```

- By default, clusters 20k papers (can be changed in code)
- Outputs `cluster_results.json`

### Step 3: Bloom Filter Preprocessing

Run the Bloom filter n-gram analysis:

```bash
python bloom_pipeline.py preprocess
```

To test specific paper pairs:

```bash
python bloom_pipeline.py test --paper_a PAPER1_ID --paper_b PAPER2_ID
```

Example:

```bash
python bloom_pipeline.py test --paper_a 1602.05576v1.tex --paper_b 1611.05859v1.tex
```

**Output:** `bloom_candidates.json`, showing overlapping phrases such as:

```
1. a function of density temperature and electron fraction
2. is a function of density temperature and electron
3. to the viability of the neutrino driven mechanism
```

### Step 4: Siamese BERT Inference

Apply the trained model to Bloom candidates:

```bash
./run_plagiarism_weights.sh
```

This uses `inference_pipeline.py` and the model checkpoint to detect semantic overlap.

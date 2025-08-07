---

````markdown
# 🧠 Plagiarism Detection Pipeline

This repository contains a complete pipeline for detecting potential plagiarism in LaTeX academic papers using Bloom Filters, Siamese BERT, and clustering techniques.

---

## 🚀 Training

We moved the training phase to a compute cluster by:
- Predefining a corpus dataset on a local machine,
- Uploading it to the cluster,
- Creating a dedicated GPU-enabled training environment.

### 📁 Required Files in the Cluster Home Folder (`~/`)

Ensure the following files are present in your **home folder** on the cluster:

1. `start_training.sh` — for environment setup and GPU access  
2. `setup_gpu_training.sh`  
3. `plagiarism_detector.py`  
4. `run_plagiarism_detection.sh`  
5. `train_pairs.json` — generated locally by running `run_plagiarism_detection.sh` (stop after corpus creation)  
6. `val_pairs.json` — same as above

> 💡 You don't need to train locally. Run the script locally only to generate the corpus, then stop it.

---

### 🏋️ Start Training

From the **login node**:

```bash
bash ~/start_training.sh
````

Then on the GPU node:

```bash
bash ./run_plagiarism_detection.sh --train_json train_pairs.json --val_json val_pairs.json --epochs 5 --output_dir /tmp/training_work
```

---

### 📦 Save Training Outputs

After training, copy outputs from `/tmp` (which is wiped between jobs):

```bash
cp /tmp/training_work/best_siamese_bert.pth ~/
cp /tmp/training_work/training_history.png ~/
```

Then:

```bash
exit
```

---

## 🔍 Inference

From the login node:

```bash
bash ~/start_inference.sh
```

This copies required files into `/tmp`:

### Included Scripts:

* `paper_counter.py` — counts available papers
* `tex_clustering.py` — clusters papers using HDBSCAN
* `bloom_pipeline.py` — Bloom filter for n-gram similarity
* `inference_pipeline.py` — semantic similarity via Siamese BERT
* `run_plagiarism_weights.sh` — executes the inference pipeline
* `best_siamese_bert.pth` — trained model weights

---

## 📊 Analysis & Inference Steps

### Step 1: Count Available Papers

```bash
python paper_counter.py
```

---

### Step 2: Clustering with HDBSCAN

```bash
python tex_clustering.py
```

* Clusters 20,000 papers by default
* Configurable via the `max_papers` variable in the script
* Outputs: `cluster_results.json`

---

### Step 3: Bloom Filter Preprocessing

Run Bloom filter across all papers:

```bash
python bloom_pipeline.py preprocess
```

Check specific paper pairs:

```bash
python bloom_pipeline.py test --paper_a PAPER1_ID --paper_b PAPER2_ID
```

Example:

```bash
python bloom_pipeline.py test --paper_a 1602.05576v1.tex --paper_b 1611.05859v1.tex
```

**Output:** `bloom_candidates.json`

Example of overlapping n-grams:

```
1. a function of density temperature and electron fraction
2. is a function of density temperature and electron
3. to the viability of the neutrino driven mechanism
```

---

### Step 4: Run Siamese BERT on Bloom Candidates

Apply the trained model weights:

```bash
./run_plagiarism_weights.sh
```

This uses:

* `inference_pipeline.py`
* `best_siamese_bert.pth`

To generate plagiarism scores for candidate paper pairs.

---

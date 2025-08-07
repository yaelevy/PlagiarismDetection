# 🧠 Plagiarism Detection Pipeline

⚠️ **BETA VERSION - UNDER ACTIVE DEVELOPMENT**  

This pipeline implements a multi-stage approach to detect plagiarism in academic papers from a large corpus of arXiv documents. The system begins by training a Siamese BERT model on the PAN Plagiarism Dataset, which learns to identify semantic similarities between text pairs that indicate potential plagiarism (for more info on PAN plagiarism dataset see appendix). Once trained, the pipeline processes approximately 1 million LaTeX papers from arXiv; First, HDBSCAN clustering groups similar papers together, based on abstract, to reduce computational overhead and focus analysis on potentially related documents within each cluster. Next, Bloom filters perform efficient n-gram analysis to identify papers with significant textual overlap within each cluster, generating a set of candidate paper pairs with shared phrases and potentially suspicious similarities. These Bloom filter candidates are then passed through the trained Siamese BERT model for semantic similarity analysis, which can detect more nuanced forms of plagiarism including paraphrasing and conceptual duplication that simple text matching might miss. The entire pipeline is designed to scale efficiently using HUJI cluster, ultimately producing a ranked list of paper pairs with high plagiarism likelihood scores for further human review.

## 🚀 Training

## 📁 Files Required in Your Home Folder on the HUJI Cluster

Ensure the following files exist in your **home folder** (`~/`) on the cluster:

1. `start_training.sh` — for environment setup and GPU access
2. `setup_gpu_training.sh`
3. `plagiarism_detector.py`
4. `run_plagiarism_detection.sh`
5. `train_pairs.json` — generated locally by running `run_plagiarism_detection.sh` (only the corpus creation stage)
6. `val_pairs.json` — same as above

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

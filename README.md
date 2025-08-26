#  Plagiarism Detection Pipeline

⚠️ **BETA VERSION - UNDER ACTIVE DEVELOPMENT**  

This pipeline implements a multi-stage approach to detect plagiarism in academic papers from a large corpus of arXiv documents. The system begins by training a Siamese BERT model on the PAN Plagiarism Dataset, which learns to identify semantic similarities between text pairs that indicate potential plagiarism (for more info on PAN plagiarism dataset see appendix). Once trained, the pipeline processes approximately 1 million LaTeX papers from arXiv; First, HDBSCAN clustering groups similar papers together, based on abstract, to reduce computational overhead and focus analysis on potentially related documents within each cluster. Next, Bloom filters perform efficient n-gram analysis to identify papers with significant textual overlap within each cluster, generating a set of candidate paper pairs with shared phrases and potentially suspicious similarities. These Bloom filter candidates are then passed through the trained Siamese BERT model for semantic similarity analysis, which can detect more nuanced forms of plagiarism including paraphrasing and conceptual duplication that simple text matching might miss. The entire pipeline is designed to scale efficiently using HUJI cluster, ultimately producing a ranked list of paper pairs with high plagiarism likelihood scores for further human review.

<img width="999" height="474" alt="צילום מסך 2025-08-26 ב-15 51 06" src="https://github.com/user-attachments/assets/d0e1d351-cf76-4896-b7aa-4567670e3d3a" />



Siamese BERT architecture:

<p align="center">
  <img width="765" height="418" alt="צילום מסך 2025-08-26 ב-15 50 56" src="https://github.com/user-attachments/assets/11944e49-0409-4263-be40-39b6dc2476e5" />
</p>





##  Training

## 📁 Files Required in Your Home Folder on the HUJI Cluster

Ensure the following files exist in your **home folder** (`~/`) on the cluster:

1. `start_training.sh` — for environment setup and GPU access
2. `setup_gpu_training.sh`
3. `plagiarism_detector.py`
4. `run_plagiarism_detection.sh`
5. `train_pairs.json` — generated locally by running `run_plagiarism_detection.sh` (only the corpus creation stage)
6. `val_pairs.json` — same as above

##  Start Training

From the **login node**:

```bash
bash ~/start_training.sh
```

Then on the GPU node:

```bash
bash ./run_plagiarism_detection.sh --train_json train_pairs.json --val_json val_pairs.json --epochs 5 --output_dir /tmp/training_work
```

##  Save Training Outputs

After training (since `/tmp` is volatile), copy the outputs:

```bash
cp /tmp/training_work/best_siamese_bert.pth ~/
cp /tmp/training_work/training_history.png ~/
```

Then exit the training node:

```bash
exit
```

##  Inference

From the login node:

```bash
bash ~/start_inference.sh
```

This copies necessary scripts and models into `/tmp` for fast access.

###  Inference Files

- `paper_counter.py` — counts available papers
- `tex_clustering.py` — clusters papers using HDBSCAN
- `bloom_pipeline.py` — Bloom filter preprocessing
- `inference_pipeline.py` — runs inference with Siamese BERT
- `run_plagiarism_weights.sh` — script to apply the trained weights
- `best_siamese_bert.pth` — trained model weights
- `process_bloom_candidates.py`
- `plagiarism_detector.py`



##  Analysis Steps

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

**Output:** `bloom_candidates.json`

To test specific paper pairs:


```bash
python bloom_pipeline.py test --paper_a 1602.05576v1.tex --paper_b 1611.05859v1.tex
```

Then we can also generate a JSON with all of the overlapping instances from candidate pairs using 

```bash
python process_bloom_candidates.py
```

**Output:** `bloom_overlap_results.json.json`


### Step 4: Siamese BERT Inference

Apply the trained model to Bloom candidates from the bloom_overlap_results.json:

```bash
bash run_plagiarism_weights.sh --max_pages 50 -t 0.8 
```
This uses `inference_pipeline.py` and the model trained weights to detect semantic overlap between paragraphs.

we chose to include papers with maximum of 50 pages and adjusted the threshold of similarity to be 0.8 (meaning pairs with similarity > 0.8 will be flagged as potential for plagiarism).

**Output:** `inference_results_{int(time.time())}.json`


## Appendix: PAN Plagiarism Dataset

### Building Ground Truth Dataset for Plagiarism Instances

The training data for this pipeline comes from the PAN Plagiarism Dataset, which provides manually annotated plagiarism instances for supervised learning. This part was done localy (not on the cluster).

#### Dataset Setup

1. **Download PAN-Plagiarism files** from the official repository:
   ```
   https://zenodo.org/records/3250095
   ```

2. **Merge external tasks** from both downloaded files into a single directory structure, consolidating all source and suspicious document instances in the external folder.

#### Extracting Plagiarism Mappings

To build the ground truth dataset, we created a Unix script that processes XML annotation files and extracts exact plagiarism passages with their corresponding offsets and metadata.

Run the extraction script:

```bash
./extract_all_parts.sh > all_plagiarism_mappings.csv
```

#### Viewing Specific Plagiarism Instances

To examine actual plagiarized text passages, place the following helper scripts in the suspicious directory and make them executable:

- `process_csv_final.sh`
- `extract_passages_final.sh` 
- `find_document_by_number.sh`

```bash
chmod +x find_document_by_number.sh extract_passages_final.sh process_csv_final.sh
source process_csv_final.sh
```

**Example Usage:**
```bash
corpus_path="/path/to/pan-plagiarism-corpus-2011-1/external-detection-corpus"
csv_file="all_plagiarism_mappings.csv"

# Extract a specific plagiarism instance (e.g., row 1)
extract_row 1 "$csv_file" "$corpus_path"
```

This process creates a comprehensive dataset of plagiarism instances with precise text offsets, enabling the Siamese BERT model to learn from real-world plagiarism patterns including various obfuscation techniques like paraphrasing, translation, and structural modifications.

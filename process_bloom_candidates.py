import json
import subprocess
import re

# Load candidates
with open('bloom_candidates.json', 'r') as f:
    candidates = json.load(f)

results = []

for candidate in candidates:
    # Extract tex filenames
    paper_a = candidate['file_a'].split('/')[-1]
    paper_b = candidate['file_b'].split('/')[-1]

    # Run bloom test
    cmd = ["python", "bloom_pipeline.py", "test", "--paper_a", paper_a, "--paper_b", paper_b]
    output = subprocess.run(cmd, capture_output=True, text=True).stdout

    # Extract key info
    actual_overlaps = int(re.search(r'Actual overlapping n-grams: (\d+)', output).group(1))
    signal_strength = int(re.search(r'Bloom signal strength: (\d+)', output).group(1))
    overlapping_ngrams = re.findall(r'\d+\. (.+)', output)

    # Store result
    results.append({
        'paper_a': paper_a,
        'paper_b': paper_b,
        'actual_overlaps': actual_overlaps,
        'signal_strength': signal_strength,
        'overlapping_ngrams': overlapping_ngrams,
        'cluster_id': candidate['cluster_id'],
        'authors_a': candidate['authors_a'],
        'authors_b': candidate['authors_b']
    })

    print(f"Processed {paper_a} vs {paper_b}: {actual_overlaps} overlaps")

# Save results
with open('bloom_overlap_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Done! Processed {len(results)} pairs.")
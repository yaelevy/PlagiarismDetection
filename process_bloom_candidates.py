import json
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import os


def process_candidate(candidate):
    """Process a single candidate pair"""
    paper_a = candidate['file_a'].split('/')[-1]
    paper_b = candidate['file_b'].split('/')[-1]

    cmd = ["python", "bloom_pipeline.py", "test", "--paper_a", paper_a, "--paper_b", paper_b]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout

        actual_overlaps = int(re.search(r'Actual overlapping n-grams: (\d+)', output).group(1))
        signal_strength = int(re.search(r'Bloom signal strength: (\d+)', output).group(1))
        overlapping_ngrams = re.findall(r'\d+\. (.+)', output)

        return {
            'file_a': candidate['file_a'],
            'file_b': candidate['file_b'],
            'actual_overlaps': actual_overlaps,
            'signal_strength': signal_strength,
            'overlapping_ngrams': overlapping_ngrams,
            'cluster_id': candidate['cluster_id'],
            'authors_a': candidate['authors_a'],
            'authors_b': candidate['authors_b']
        }
    except Exception as e:
        print(f"Error processing {paper_a} vs {paper_b}: {e}")
        return None


# Load candidates
with open('bloom_candidates.json', 'r') as f:
    candidates = json.load(f)

results = []
max_workers = min(os.cpu_count(), 8)  # Adjust based on your system

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit all jobs
    future_to_candidate = {executor.submit(process_candidate, candidate): candidate
                           for candidate in candidates}

    # Collect results as they complete
    for i, future in enumerate(as_completed(future_to_candidate)):
        result = future.result()
        if result:
            results.append(result)

        if i % 100 == 0:
            print(f"{i + 1}/{len(candidates)} completed")

# Sort and save
results.sort(key=lambda x: x['actual_overlaps'], reverse=True)
with open('bloom_overlap_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Done! Processed {len(results)} pairs.")

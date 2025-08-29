#!/usr/bin/env python3
"""
Unified Bloom Filter Pipeline for Plagiarism Detection
Optimized for speed with caching and parallel comparisons
"""

import os
import re
import json
import hashlib
import argparse
import time
import pickle
from itertools import combinations
from multiprocessing import Pool, cpu_count
from bitarray import bitarray
import numpy as np

# -----------------------
# GLOBAL CONFIGURATION
# -----------------------
BLOOM_CONFIG = {
    'capacity': 50000,
    'error_rate': 0.001,
    'ngram_size': 8,
    'threshold': 1500,
    'high_priority_threshold': 2000,
    'cache_dir': "bloom_cache"
}

os.makedirs(BLOOM_CONFIG['cache_dir'], exist_ok=True)

# -----------------------
# BLOOM FILTER CLASS
# -----------------------
class BloomFilter:
    def __init__(self, capacity=None, error_rate=None):
        self.capacity = capacity or BLOOM_CONFIG['capacity']
        self.error_rate = error_rate or BLOOM_CONFIG['error_rate']
        size_multiplier = 3
        self.size = int(-self.capacity * np.log(self.error_rate) / (np.log(2) ** 2)) * size_multiplier
        self.hash_count = min(5, int((self.size / self.capacity) * np.log(2)))
        self.bit_array = bitarray(self.size)
        self.bit_array.setall(0)

    def _hash(self, item, seed):
        return int(hashlib.md5((str(item) + str(seed)).encode()).hexdigest(), 16) % self.size

    def add(self, item):
        for i in range(self.hash_count):
            self.bit_array[self._hash(item, i)] = 1

    def check_overlap(self, other, threshold=None):
        threshold = threshold or BLOOM_CONFIG['threshold']
        count = (self.bit_array & other.bit_array).count()
        return count >= threshold, count

    def save(self, path):
        with open(path, "wb") as f:
            self.bit_array.tofile(f)

    @staticmethod
    def load(path, size=None):
        bf = BloomFilter()
        if os.path.exists(path):
            with open(path, "rb") as f:
                bf.bit_array.fromfile(f)
        return bf

# -----------------------
# TEXT PROCESSING
# -----------------------
def extract_text(tex_file):
    """Extract text content optimized for recurring phrase detection."""
    try:
        with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Remove LaTeX comments
        content = re.sub(r'^%.*$', '', content, flags=re.MULTILINE)
        # Extract body
        m = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', content, re.DOTALL)
        if m:
            content = m.group(1)
        # Remove environments
        patterns = [r'\\begin\{.*?\}.*?\\end\{.*?\}', r'\$\$.*?\$\$', r'\$.*?\$']
        for pat in patterns:
            content = re.sub(pat, ' ', content, flags=re.DOTALL)
        # Remove commands but preserve citations
        content = re.sub(r'\\cite[tp]?\{[^}]*\}', '[CITE]', content)
        content = re.sub(r'\\ref\{[^}]*\}', '[REF]', content)
        content = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{[^}]*\}', '', content)
        content = re.sub(r'\\[a-zA-Z]+\*?', '', content)
        content = re.sub(r'[{}~]', ' ', content)
        content = re.sub(r'\s+', ' ', content)
        return content.strip().lower()
    except Exception as e:
        print(f"Error reading {tex_file}: {e}")
        return ""

def generate_ngrams(text, n=None):
    n = n or BLOOM_CONFIG['ngram_size']
    words = re.findall(r'\b[a-zA-Z]+\b|\[CITE\]|\[REF\]', text)
    if len(words) < n:
        return []
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

# -----------------------
# BLOOM FILTER CACHING
# -----------------------
def load_or_create_bloom(file_path):
    cache_path = os.path.join(BLOOM_CONFIG['cache_dir'], os.path.basename(file_path) + ".bf")
    if os.path.exists(cache_path):
        bf = BloomFilter.load(cache_path)
        return bf
    text = extract_text(file_path)
    ngrams = generate_ngrams(text)
    if len(ngrams) < 10:
        return None
    bf = BloomFilter()
    for ng in ngrams:
        bf.add(ng)
    bf.save(cache_path)
    return bf

# -----------------------
# AUTHOR EXTRACTION
# -----------------------
def clean_author(name):
    if not name:
        return None
    name = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower() if len(name.split()) >= 2 else None

def extract_authors(tex_file):
    try:
        with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        authors = []
        for m in re.findall(r'\\author\s*\{([^}]*)\}', content):
            parts = re.split(r'\\and|,|\\\\', m)
            for p in parts:
                c = clean_author(p)
                if c and c not in authors:
                    authors.append(c)
        return authors
    except:
        return []

def has_common_authors(file_a, file_b):
    a = set(extract_authors(file_a))
    b = set(extract_authors(file_b))
    common = a.intersection(b)
    return bool(common), common

# -----------------------
# CLUSTER PREPROCESSING
# -----------------------
def compare_pair(args):
    file_a, file_b, threshold = args
    bf_a = load_or_create_bloom(file_a)
    bf_b = load_or_create_bloom(file_b)
    if not bf_a or not bf_b:
        return None
    overlap_flag, count = bf_a.check_overlap(bf_b, threshold=threshold)
    if not overlap_flag:
        return None
    same_authors, common = has_common_authors(file_a, file_b)
    return {
        'file_a': file_a,
        'file_b': file_b,
        'signal_strength': count,
        'priority': 'high' if count > BLOOM_CONFIG['high_priority_threshold'] else 'medium',
        'common_authors': list(common),
        'same_authors': same_authors
    }

def preprocess_clusters():
    if not os.path.exists("cluster_results.json"):
        print("cluster_results.json not found!")
        return
    with open("cluster_results.json", 'r') as f:
        clusters = json.load(f)
    all_candidates = []
    for cid, files in clusters.items():
        if len(files) < 2:
            continue
        pairs = list(combinations(files, 2))
        with Pool(cpu_count()) as pool:
            results = pool.map(compare_pair, [(a,b,BLOOM_CONFIG['threshold']) for a,b in pairs])
        results = [r for r in results if r and not r['same_authors']]
        for r in results:
            r['cluster_id'] = cid
        all_candidates.extend(results)
    # Save results
    with open("bloom_candidates.json", "w") as f:
        json.dump(all_candidates, f, indent=2)
    print(f"Preprocessing complete. {len(all_candidates)} candidate pairs saved.")
    return all_candidates

# -----------------------
# TEST MODE
# -----------------------
def test_pair(file_a, file_b):
    bf_a = load_or_create_bloom(file_a)
    bf_b = load_or_create_bloom(file_b)
    if not bf_a or not bf_b:
        print("One of the files is too short or invalid.")
        return
    overlap, count = bf_a.check_overlap(bf_b)
    same_authors, common = has_common_authors(file_a, file_b)
    print(f"\nTest result for {os.path.basename(file_a)} <-> {os.path.basename(file_b)}")
    print(f"Overlap signal: {count}")
    print(f"Same authors: {same_authors}, Common: {common}")
    print(f"Flagged as candidate: {overlap and not same_authors}")

# -----------------------
# MAIN
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Bloom Filter Pipeline")
    parser.add_argument('mode', choices=['preprocess','test'])
    parser.add_argument('--paper_a')
    parser.add_argument('--paper_b')
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--show-config', action='store_true')
    args = parser.parse_args()

    if args.show_config:
        print(json.dumps(BLOOM_CONFIG, indent=2))
        return

    if args.threshold:
        BLOOM_CONFIG['threshold'] = args.threshold

    if args.mode == 'preprocess':
        preprocess_clusters()
    elif args.mode == 'test':
        if not args.paper_a or not args.paper_b:
            print("Provide --paper_a and --paper_b for test mode")
            return
        test_pair(args.paper_a, args.paper_b)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified Bloom Filter Pipeline for Plagiarism Detection
Ensures consistent configuration between preprocessing and testing
"""

import os
import re
import json
import hashlib
import numpy as np
import argparse
import time
from collections import defaultdict, Counter

# GLOBAL CONFIGURATION - Single source of truth
BLOOM_CONFIG = {
    'capacity': 50000,
    'error_rate': 0.001,
    'ngram_size': 8,
    'threshold': 1500,
    'high_priority_threshold': 2000
}


class BloomFilter:
    def __init__(self, capacity=None, error_rate=None):
        """Initialize bloom filter with global config"""
        self.capacity = capacity or BLOOM_CONFIG['capacity']
        self.error_rate = error_rate or BLOOM_CONFIG['error_rate']

        # Calculate optimal bloom filter size and number of hash functions
        size_multiplier = 3  # Double the calculated size
        self.size = int(-capacity * np.log(error_rate) / (np.log(2) ** 2)) * size_multiplier
        self.hash_count = min(5, int((self.size / self.capacity) * np.log(2)))

        # Initialize bit array
        self.bit_array = [0] * self.size

        #print(f"Bloom filter: size={self.size}, hash_functions={self.hash_count}")

    def _hash(self, item, seed):
        """Generate hash for an item with given seed"""
        hash_obj = hashlib.md5((str(item) + str(seed)).encode())
        return int(hash_obj.hexdigest(), 16) % self.size

    def add(self, item):
        """Add an item to the bloom filter"""
        for i in range(self.hash_count):
            hash_val = self._hash(item, i)
            self.bit_array[hash_val] = 1

    def check_for_recurring_phrases(self, other_bloom, threshold=None):
        """
        Check if two documents likely share recurring phrases
        Returns (has_recurring_phrases, overlap_signal_strength)
        """
        threshold = threshold or BLOOM_CONFIG['threshold']

        if len(self.bit_array) != len(other_bloom.bit_array):
            raise ValueError("Bloom filters must be same size for comparison")

        # Count overlapping 1 bits (signal + noise)
        overlap_count = sum(1 for i in range(len(self.bit_array))
                            if self.bit_array[i] == 1 and other_bloom.bit_array[i] == 1)

        # Use global threshold
        has_recurring = overlap_count >= threshold

        return has_recurring, overlap_count


def extract_content_for_phrases(tex_file_path):
    """Extract text content optimized for detecting recurring phrases"""
    try:
        with open(tex_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            tex_content = f.read()

        # Remove comments
        tex_content = re.sub(r'^%.*$', '', tex_content, flags=re.MULTILINE)

        # Extract document body
        body_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}',
                               tex_content, re.DOTALL)
        if body_match:
            tex_content = body_match.group(1)

        # Remove non-textual content
        environments_to_remove = [
            r'\\begin\{table\}.*?\\end\{table\}',
            r'\\begin\{figure\}.*?\\end\{figure\}',
            r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}',
            r'\\begin\{align\*?\}.*?\\end\{align\*?\}',
            r'\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}',
            r'\\begin\{tabular\}.*?\\end\{tabular\}',
            r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
            r'\\begin\{algorithm\}.*?\\end\{algorithm\}',
            r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}',
            r'\\begin\{verbatim\}.*?\\end\{verbatim\}',
        ]

        for env_pattern in environments_to_remove:
            tex_content = re.sub(env_pattern, ' ', tex_content, flags=re.DOTALL)

        # Remove math expressions
        tex_content = re.sub(r'\$\$.*?\$\$', ' ', tex_content, flags=re.DOTALL)
        tex_content = re.sub(r'\$.*?\$', ' ', tex_content)

        # Preserve structure by keeping citations/references as tokens
        tex_content = re.sub(r'\\cite[tp]?\{[^}]*\}', '[CITE]', tex_content)
        tex_content = re.sub(r'\\ref\{[^}]*\}', '[REF]', tex_content)

        # Remove LaTeX formatting but preserve content
        tex_content = re.sub(r'\\textbf\{([^}]*)\}', r'\1', tex_content)
        tex_content = re.sub(r'\\textit\{([^}]*)\}', r'\1', tex_content)
        tex_content = re.sub(r'\\emph\{([^}]*)\}', r'\1', tex_content)
        tex_content = re.sub(r'\\text\{([^}]*)\}', r'\1', tex_content)

        # Remove other LaTeX commands
        tex_content = re.sub(r'\\[a-zA-Z]+\*?\[[^\]]*\]\{[^}]*\}', '', tex_content)
        tex_content = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', tex_content)
        tex_content = re.sub(r'\\[a-zA-Z]+\*?', '', tex_content)

        # Clean up
        tex_content = re.sub(r'[{}]', '', tex_content)
        tex_content = re.sub(r'~', ' ', tex_content)
        tex_content = re.sub(r'\s+', ' ', tex_content)

        return tex_content.strip().lower()

    except Exception as e:
        print(f"Error reading {tex_file_path}: {e}")
        return ""


def generate_phrase_ngrams(text, n=None):
    """
    Generate n-grams using global configuration
    """
    n = n or BLOOM_CONFIG['ngram_size']

    if not text:
        return []

    # Extract words, keep meaningful tokens
    words = re.findall(r'\b[a-zA-Z]+\b|\[CITE\]|\[REF\]', text.lower())

    if len(words) < n:
        return []

    # Generate overlapping n-grams
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)

    return ngrams


def create_phrase_detection_bloom(file_path):
    """Create bloom filter using global configuration"""
    #print(f"Processing: {os.path.basename(file_path)}")

    # Extract content
    text = extract_content_for_phrases(file_path)

    if len(text) < 100:
        print(f"  Skipping - content too short")
        return None

    # Generate n-grams using global config
    ngrams = generate_phrase_ngrams(text)

    if len(ngrams) < 10:
        print(f"  Skipping - too few phrases ({len(ngrams)})")
        return None

    #print(f"  Generated {len(ngrams)} {BLOOM_CONFIG['ngram_size']}-grams")

    # Create bloom filter with global config
    bloom = BloomFilter(capacity=50000, error_rate=0.001)

    # Add n-grams to bloom filter
    for ngram in ngrams:
        bloom.add(ngram)

    return bloom


def find_complete_author_block(content, partial_match):
    """Find complete author block when regex truncates due to nested braces"""
    author_start = content.find('\\author{')
    if author_start == -1:
        return None

    # Find matching closing brace
    brace_count = 0
    start_pos = author_start + 8  # After '\author{'

    for i, char in enumerate(content[start_pos:], start_pos):
        if char == '{':
            brace_count += 1
        elif char == '}':
            if brace_count == 0:
                return content[start_pos:i]
            brace_count -= 1

    return None


def extract_complete_multiline_author(content, start_pos):
    """Extract complete multiline author block"""
    # Find the opening brace
    brace_start = content.find('{', start_pos)
    if brace_start == -1:
        return None

    # Count braces to find the matching closing brace
    brace_count = 1
    pos = brace_start + 1

    while pos < len(content) and brace_count > 0:
        char = content[pos]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        pos += 1

    if brace_count == 0:
        return content[brace_start + 1:pos - 1]  # Extract content between braces

    return None


def extract_authors_from_tex(tex_file_path):
    """Extract author names from LaTeX file with improved parsing"""
    try:
        with open(tex_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        authors = []

        # Pattern 1: \author[short]{full name}
        pattern1 = r'\\author\s*\[([^\]]*)\]\s*\{([^}]+)\}'
        matches1 = re.findall(pattern1, content, re.IGNORECASE)
        for short_name, full_name in matches1:
            # FIXED: Split by \and and \\ BEFORE cleaning
            if '\\and' in full_name or '\\\\' in full_name:
                # Split by \and first
                author_parts = re.split(r'\\and', full_name)
                for part in author_parts:
                    # Then split by line breaks
                    author_lines = re.split(r'\\\\', part)
                    for author_line in author_lines:
                        clean_name = clean_author_name(author_line)
                        if clean_name and clean_name not in authors:
                            authors.append(clean_name)
            else:
                clean_name = clean_author_name(full_name)
                if clean_name and clean_name not in authors:
                    authors.append(clean_name)

        # Pattern 2: Standard \author{name}
        pattern2 = r'\\author\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'

        matches2 = re.findall(pattern2, content, re.IGNORECASE)
        for match in matches2:
            if len(match) < 10 and ("\\'" in match or match.endswith('e')):
                # This looks like a truncated match, find the complete author block
                author_start = content.find('\\author{')
                if author_start != -1:
                    # Find the complete multiline author block
                    complete_match = extract_complete_multiline_author(content, author_start)
                    if complete_match:
                        match = complete_match

            # Skip if already captured by pattern1
            if not re.search(r'\\author\[[^\]]*\]\{' + re.escape(match) + r'\}', content):
                # FIXED: Handle multiple authors in one command BEFORE cleaning
                if '\\and' in match or '\\\\' in match or ',' in match:  # <-- ADD ", ',' in match"
                    # Split by \and first
                    author_parts = re.split(r'\\and', match)
                    for part in author_parts:
                        # Then split by line breaks
                        author_lines = re.split(r'\\\\', part)
                        for author_line in author_lines:
                            # NEW: Also split by comma for comma-separated authors
                            comma_parts = re.split(r',(?![^{]*})', author_line)  # Split on comma but not inside braces
                            for comma_part in comma_parts:
                                clean_name = clean_author_name(comma_part)
                                if clean_name and clean_name not in authors:
                                    authors.append(clean_name)
                else:
                    # Single author
                    clean_name = clean_author_name(match)
                    if clean_name and clean_name not in authors:
                        authors.append(clean_name)

        # Pattern 3: \name{...} format (used by some conferences)
        pattern3 = r'\\name\s*\{([^}]+)\}'
        matches3 = re.findall(pattern3, content, re.IGNORECASE)
        for match in matches3:
            # Handle multiple authors in \name{Author1, Author2, Author3} format
            if ',' in match:
                author_parts = match.split(',')
                for part in author_parts:
                    # Remove affiliation markers like $^{\ast}$, $^{\dagger}$
                    clean_part = re.sub(r'\$\^[^$]*\$', '', part.strip())
                    clean_name = clean_author_name(clean_part)
                    if clean_name and clean_name not in authors:
                        authors.append(clean_name)
            else:
                # Single author in \name
                clean_part = re.sub(r'\$\^[^$]*\$', '', match.strip())
                clean_name = clean_author_name(clean_part)
                if clean_name and clean_name not in authors:
                    authors.append(clean_name)

        # Pattern 4: Multiline \author{...} with nested braces (LLNCS format)
        # Use a more sophisticated approach to match balanced braces
        pattern4 = r'\\author\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}'
        matches4 = re.findall(pattern4, content, re.DOTALL | re.IGNORECASE)
        for match in matches4:
            if match not in [m[1] if isinstance(m, tuple) else m for m in
                             matches1 + [m for m in matches2]]:  # Avoid duplicates
                # Handle \and separator and clean institutional markup
                if '\\and' in match:
                    author_parts = re.split(r'\\and', match)
                    for part in author_parts:
                        # Remove institutional markup like \inst{1}, \thanks{...}
                        clean_part = re.sub(r'\\inst\{[^}]*\}', '', part)
                        clean_part = re.sub(r'\\thanks\{[^}]*\}', '', clean_part)
                        clean_name = clean_author_name(clean_part)
                        if clean_name and clean_name not in authors:
                            authors.append(clean_name)

        # ADDITIONAL FIX: Split any remaining compound names with "and" in them
        final_authors = []
        for author in authors:
            # If an author still has "and" in the middle, split it
            if ' and ' in author and not any(word in author.lower() for word in ['supported', 'university', 'department']):
                parts = author.split(' and ')
                for part in parts:
                    clean_part = clean_author_name(part)
                    if clean_part and clean_part not in final_authors:
                        final_authors.append(clean_part)
            else:
                if author not in final_authors:
                    final_authors.append(author)

        return final_authors

    except Exception as e:
        print(f"Error reading {tex_file_path}: {e}")
        return []



def clean_author_name(name):
    """Clean LaTeX formatting from author names with better handling"""
    if not name or not isinstance(name, str):
        return None

    # Remove LaTeX accent commands
    name = re.sub(r'\\[v"\'`\^~c]\{([^}])\}', r'\1', name)

    # Remove LaTeX line breaks and formatting
    name = re.sub(r'\\\\+', '', name)
    name = re.sub(r'\\&', '&', name)

    # Remove superscripts, footnotes, thanks
    name = re.sub(r'\$\^?\{?[0-9]+\}?\$', '', name)
    name = re.sub(r'\^[0-9]+', '', name)
    name = re.sub(r'\\thanks\{[^}]*\}', '', name)
    name = re.sub(r'\\footnote\{[^}]*\}', '', name)

    # Remove other LaTeX commands
    name = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', name)
    name = re.sub(r'\\[a-zA-Z]+', '', name)

    # Remove institutional info (more comprehensive)
    name = re.sub(r'supported by.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'university.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'department.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'institute.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'school.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'college.*', '', name, flags=re.IGNORECASE)

    # Remove email addresses
    name = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', name)

    # Remove URLs
    name = re.sub(r'https?://[^\s]+', '', name)

    # Remove affiliations in parentheses
    name = re.sub(r'\([^)]*(?:university|institute|dept|department)[^)]*\)', '', name, flags=re.IGNORECASE)

    # Clean up punctuation and spacing
    name = re.sub(r'[{}]', '', name)
    name = re.sub(r'[,;]+$', '', name)
    name = re.sub(r'^[,;]+', '', name)
    name = re.sub(r'\s*,\s*$', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()

    # Remove superscripts, footnotes, thanks
    name = re.sub(r'\$\^?\{?[0-9]+\}?\$', '', name)
    name = re.sub(r'\^[0-9]+', '', name)
    name = re.sub(r'\\thanks\{[^}]*\}', '', name)
    name = re.sub(r'\\footnote\{[^}]*\}', '', name)

    # ADD THESE NEW LINES HERE:
    # Remove LaTeX superscript notation
    name = re.sub(r'\$\^\{?[0-9]+\}?\$', '', name)
    name = re.sub(r'\$\^[0-9]+', '', name)
    # Remove LaTeX line breaks that aren't being caught
    name = re.sub(r'\\\\', ' ', name)

    # Only keep reasonable names (more strict validation)
    if (len(name) > 3 and len(name) < 60 and
            len(name.split()) >= 2 and
            not re.search(r'\d{4}', name) and  # No years
            not re.search(r'(?:abstract|introduction|conclusion|references)', name, re.IGNORECASE)):
        return name.lower()

    return None



def has_common_authors(file_a, file_b):
    """Check if two papers share common authors with improved matching"""
    authors_a = extract_authors_from_tex(file_a)
    authors_b = extract_authors_from_tex(file_b)

    # Debug: print extracted authors
    # print(f"    Authors A ({os.path.basename(file_a)}): {authors_a}")
    # print(f"    Authors B ({os.path.basename(file_b)}): {authors_b}")

    if not authors_a or not authors_b:
        # print(f"    → No authors found in one or both files")
        return False, set()

    # QUICK FIX: Split any remaining compound author strings
    split_authors_a = []
    for author in authors_a:
        # Split on common separators that weren't caught during extraction
        parts = re.split(r'\s*,\s*(?![^(]*\))', author)  # Split on comma but not inside parentheses
        for part in parts:
            clean_part = clean_author_name(part)
            if clean_part and clean_part not in split_authors_a:
                split_authors_a.append(clean_part)

    split_authors_b = []
    for author in authors_b:
        parts = re.split(r'\s*,\s*(?![^(]*\))', author)
        for part in parts:
            clean_part = clean_author_name(part)
            if clean_part and clean_part not in split_authors_b:
                split_authors_b.append(clean_part)

    # Use the split versions
    authors_a = split_authors_a
    authors_b = split_authors_b

    # FIXED: Exact matching first - convert to sets for proper intersection
    set_a = set(authors_a)
    set_b = set(authors_b)
    exact_common = set_a.intersection(set_b)

    if exact_common:
        # print(f"    → Exact match found: {exact_common}")
        return True, exact_common

    # Fuzzy matching for similar names
    fuzzy_matches = set()
    for author_a in authors_a:
        for author_b in authors_b:
            if fuzzy_author_match(author_a, author_b):
                fuzzy_matches.add(f"{author_a} ≈ {author_b}")

    if fuzzy_matches:
        # print(f"    → Fuzzy match found: {fuzzy_matches}")
        return True, fuzzy_matches

    # print(f"    → No common authors found")
    return False, set()



def fuzzy_author_match(author1, author2):
    """Check if two author strings likely refer to the same person"""
    if not author1 or not author2:
        return False

    # QUICK FIX: Add exact match check first
    if author1.lower().strip() == author2.lower().strip():
        return True

    words1 = author1.lower().split()
    words2 = author2.lower().split()

    # Remove common words
    common_words = {'and', 'et', 'al', 'with', 'by', 'the', 'corresponding', 'author', 'jr', 'sr'}
    words1 = [w for w in words1 if w not in common_words]
    words2 = [w for w in words2 if w not in common_words]

    if len(words1) < 1 or len(words2) < 1:
        return False

    # NEW: Handle initials matching
    # Check if one author uses initials and the other uses full names
    for w1 in words1:
        for w2 in words2:
            # Check if w1 is an initial that matches w2's first letter
            if len(w1) == 2 and w1.endswith('.') and w2.startswith(w1[0]):
                words1_copy = words1.copy()
                words2_copy = words2.copy()
                words1_copy.remove(w1)
                words2_copy.remove(w2)
                # Check if remaining words match
                if set(words1_copy).intersection(set(words2_copy)):
                    return True
            # Check if w2 is an initial that matches w1's first letter
            elif len(w2) == 2 and w2.endswith('.') and w1.startswith(w2[0]):
                words1_copy = words1.copy()
                words2_copy = words2.copy()
                words1_copy.remove(w1)
                words2_copy.remove(w2)
                # Check if remaining words match
                if set(words1_copy).intersection(set(words2_copy)):
                    return True
            # Check single letter initials (no period)
            elif len(w1) == 1 and w2.startswith(w1):
                words1_copy = words1.copy()
                words2_copy = words2.copy()
                words1_copy.remove(w1)
                words2_copy.remove(w2)
                if set(words1_copy).intersection(set(words2_copy)):
                    return True
            elif len(w2) == 1 and w1.startswith(w2):
                words1_copy = words1.copy()
                words2_copy = words2.copy()
                words1_copy.remove(w1)
                words2_copy.remove(w2)
                if set(words1_copy).intersection(set(words2_copy)):
                    return True

    # Original word overlap matching
    words1_set = set(words1)
    words2_set = set(words2)
    overlap = words1_set.intersection(words2_set)
    smaller_set_size = min(len(words1_set), len(words2_set))

    if smaller_set_size == 1:
        return len(overlap) == 1  # Exact match for single names
    else:
        overlap_ratio = len(overlap) / smaller_set_size
        return overlap_ratio >= 0.75  # 75% overlap for multi-word names

def test_bloom_accuracy(file_a, file_b):
    """Test bloom filter accuracy against ground truth"""
    print("=" * 80)
    print("BLOOM FILTER ACCURACY TEST")
    print("=" * 80)
    print(f"Configuration: {BLOOM_CONFIG}")
    print("=" * 80)

    print(f"Testing files:")
    print(f"  A: {os.path.basename(file_a)}")
    print(f"  B: {os.path.basename(file_b)}")
    print()

    # Extract content using SAME function as preprocessing
    text_a = extract_content_for_phrases(file_a)
    text_b = extract_content_for_phrases(file_b)

    print(f"Text extracted:")
    print(f"  File A: {len(text_a)} characters")
    print(f"  File B: {len(text_b)} characters")
    print()

    # Generate n-grams using SAME function as preprocessing
    ngrams_a = generate_phrase_ngrams(text_a)
    ngrams_b = generate_phrase_ngrams(text_b)

    print(f"N-grams generated:")
    print(f"  File A: {len(ngrams_a)} {BLOOM_CONFIG['ngram_size']}-grams")
    print(f"  File B: {len(ngrams_b)} {BLOOM_CONFIG['ngram_size']}-grams")
    print()

    # Find actual overlaps (GROUND TRUTH)
    set_a = set(ngrams_a)
    set_b = set(ngrams_b)
    overlapping_ngrams = set_a.intersection(set_b)
    actual_overlaps = len(overlapping_ngrams)

    print(f"GROUND TRUTH:")
    print(f"  Actual overlapping n-grams: {actual_overlaps}")

    if actual_overlaps > 0:
        print(f"\nFirst 10 overlapping n-grams:")
        for i, ngram in enumerate(sorted(overlapping_ngrams)[:10]):
            print(f"  {i + 1}. {ngram}")
        if actual_overlaps > 10:
            print(f"  ... and {actual_overlaps - 10} more")
    print()

    # Test bloom filter using SAME configuration as preprocessing
    bloom_a = BloomFilter(capacity=BLOOM_CONFIG['capacity'], error_rate=BLOOM_CONFIG['error_rate'])
    bloom_b = BloomFilter(capacity=BLOOM_CONFIG['capacity'], error_rate=BLOOM_CONFIG['error_rate'])

    # Add n-grams
    for ngram in ngrams_a:
        bloom_a.add(ngram)

    for ngram in ngrams_b:
        bloom_b.add(ngram)

    # Test with different thresholds
    test_thresholds = [
        BLOOM_CONFIG['threshold'] // 2,  # Half threshold
        BLOOM_CONFIG['threshold'],  # Exact threshold
        BLOOM_CONFIG['high_priority_threshold'],  # High priority threshold
        BLOOM_CONFIG['threshold'] * 2  # Double threshold
    ]

    print("BLOOM FILTER RESULTS:")
    print("-" * 60)

    for threshold in test_thresholds:
        has_overlap, bloom_overlap_count = bloom_a.check_for_recurring_phrases(
            bloom_b, threshold=threshold
        )

        # Calculate accuracy
        false_positive = has_overlap and (actual_overlaps == 0)
        false_negative = not has_overlap and (actual_overlaps > 0)
        correct_positive = has_overlap and (actual_overlaps > 0)
        correct_negative = not has_overlap and (actual_overlaps == 0)

        status = "FALSE POSITIVE" if false_positive else \
            "FALSE NEGATIVE" if false_negative else \
                "CORRECT POSITIVE" if correct_positive else \
                    "CORRECT NEGATIVE"

        priority = "HIGH" if threshold == BLOOM_CONFIG['high_priority_threshold'] else \
            "CURRENT" if threshold == BLOOM_CONFIG['threshold'] else \
                "LOW" if threshold == BLOOM_CONFIG['threshold'] // 2 else "STRICT"

        print(f"Threshold: {threshold} ({priority})")
        print(f"  Bloom signal strength: {bloom_overlap_count}")
        print(f"  Flagged as candidate: {has_overlap}")
        print(f"  Result: {status}")
        print()

    # Recommendations
    print("=" * 60)
    print("ANALYSIS & RECOMMENDATIONS:")
    print("=" * 60)

    if actual_overlaps == 0:
        print("✅ GROUND TRUTH: No actual text overlap between these papers")
        print("⚠️  Any bloom filter detection is a FALSE POSITIVE")
        print("💡 These papers are topically similar but don't share recurring phrases")
        print()
        print("RECOMMENDATIONS:")
        print(f"  • Current threshold ({BLOOM_CONFIG['threshold']}) may be too low")
        print("  • Consider increasing threshold to reduce false positives")
        print("  • Or accept some false positives for better recall")
    else:
        print(f"✅ GROUND TRUTH: {actual_overlaps} actual recurring phrase overlaps")
        print("✅ Bloom filter should detect this (legitimate signal)")
        print()
        if actual_overlaps > 50:
            print("🚨 HIGH OVERLAP: This could indicate:")
            print("   • Legitimate plagiarism case")
            print("   • Same authors reusing their own content")
            print("   • Papers from same research group/topic")
        print()
        print("RECOMMENDATIONS:")
        print("  • Send this pair to your ML model for detailed analysis")
        print("  • Check for common authors first")

    return actual_overlaps, bloom_overlap_count, overlapping_ngrams


def preprocess_clusters():
    """Run preprocessing pipeline"""
    cluster_file = "cluster_results.json"
    if not os.path.exists(cluster_file):
        print(f"Error: {cluster_file} not found.")
        return

    with open(cluster_file, 'r') as f:
        cluster_data = json.load(f)

    print(f"BLOOM FILTER PREPROCESSING")
    print(f"Configuration: {BLOOM_CONFIG}")
    print(f"Loaded {len(cluster_data)} clusters")

    # Process clusters with at least 5 files
    target_clusters = []
    for cluster_id, cluster_files in cluster_data.items():
        if len(cluster_files) >= 5:
            target_clusters.append((cluster_id, cluster_files))

    print(f"Processing {len(target_clusters)} clusters (≥5 documents each)")

    all_candidates = []
    start_time = time.time()

    for cluster_id, cluster_files in target_clusters:
        candidates = preprocess_cluster_for_recurring_phrases(cluster_files, cluster_id)
        all_candidates.extend(candidates)

    end_time = time.time()
    processing_time = end_time - start_time

    # Final summary
    print(f"\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print(f"=" * 80)
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"📊 Total candidates for ML analysis: {len(all_candidates)}")

    high_priority = len([c for c in all_candidates if c['priority'] == 'high'])
    medium_priority = len([c for c in all_candidates if c['priority'] == 'medium'])

    print(f"🔥 High priority (>{BLOOM_CONFIG['high_priority_threshold']} signal): {high_priority}")
    print(f"📋 Medium priority (>{BLOOM_CONFIG['threshold']} signal): {medium_priority}")

    if all_candidates:
        # Sort by signal strength
        all_candidates.sort(key=lambda x: x['signal_strength'], reverse=True)

        print(f"\n🏆 TOP 10 candidates:")
        for i, candidate in enumerate(all_candidates[:10]):
            name_a = os.path.basename(candidate['file_a'])
            name_b = os.path.basename(candidate['file_b'])
            print(f"  {i + 1}. {name_a} <-> {name_b}")
            print(f"     Signal: {candidate['signal_strength']}, Priority: {candidate['priority']}")

        # Save results
        output_file = "bloom_candidates.json"
        with open(output_file, 'w') as f:
            json.dump(all_candidates, f, indent=2)
        print(f"\n💾 Candidates saved to {output_file}")

    return all_candidates


def preprocess_cluster_for_recurring_phrases(cluster_files, cluster_id, threshold=1500):
    """
    Preprocess a cluster to find paper pairs with potentially recurring phrases
    Returns candidates for expensive ML analysis
    """
    print(f"\n" + "=" * 60)
    print(f"PREPROCESSING Cluster {cluster_id} ({len(cluster_files)} documents)")
    print(f"Using Bloom filters as fast pre-filter for recurring phrases")
    print(f"Threshold: {threshold} overlaps")
    print(f"=" * 60)

    # Create bloom filters for all valid documents
    bloom_filters = {}
    valid_files = []

    for file_path in cluster_files:
        bloom_filter = create_phrase_detection_bloom(file_path)
        if bloom_filter is not None:
            bloom_filters[file_path] = bloom_filter
            valid_files.append(file_path)

    print(f"\nCreated bloom filters for {len(valid_files)}/{len(cluster_files)} documents")

    if len(valid_files) < 2:
        print("Not enough valid documents for comparison")
        return []

    # Fast pairwise comparison using bloom filters
    candidates_for_ml = []
    filtered_same_author = []
    filtered_no_phrases = []

    total_comparisons = len(valid_files) * (len(valid_files) - 1) // 2
    comparison_count = 0

    print(f"\nStarting {total_comparisons} fast bloom filter comparisons...")

    for i in range(len(valid_files)):
        for j in range(i + 1, len(valid_files)):
            comparison_count += 1

            if comparison_count % 100 == 0:
                print(f"  Progress: {comparison_count}/{total_comparisons} comparisons")

            file_a = valid_files[i]
            file_b = valid_files[j]

            bloom_a = bloom_filters[file_a]
            bloom_b = bloom_filters[file_b]

            # Fast bloom filter check for recurring phrases
            has_phrases, signal_strength = bloom_a.check_for_recurring_phrases(
                bloom_b, threshold=threshold
            )

            if has_phrases:
                #print(f"\n  Potential recurring phrases detected:")
                #print(f"    {os.path.basename(file_a)} <-> {os.path.basename(file_b)}")
                #print(f"    Signal strength: {signal_strength}")

                # Check for same authors (legitimate reuse) - WITH DEBUG OUTPUT
                has_common, common_authors = has_common_authors(file_a, file_b)

                if has_common:
                    #print(f"    → FILTERED: Same authors detected")
                    filtered_same_author.append({
                        'file_a': file_a,
                        'file_b': file_b,
                        'signal_strength': signal_strength,
                        'common_authors': list(common_authors),
                        'cluster_id': cluster_id
                    })
                else:
                    #print(f"    → CANDIDATE: Different authors, flagged for ML analysis")
                    # Extract authors for both files for storage
                    authors_a = extract_authors_from_tex(file_a)
                    authors_b = extract_authors_from_tex(file_b)

                    # This is a candidate for ML analysis
                    candidates_for_ml.append({
                        'file_a': file_a,
                        'file_b': file_b,
                        'signal_strength': signal_strength,
                        'cluster_id': cluster_id,
                        'priority': 'high' if signal_strength > 2000 else 'medium',
                        'authors_a': authors_a,
                        'authors_b': authors_b
                    })
            else:
                filtered_no_phrases.append({
                    'file_a': file_a,
                    'file_b': file_b,
                    'signal_strength': signal_strength,
                    'cluster_id': cluster_id
                })

    # Results summary
    print(f"\n" + "=" * 60)
    print(f"CLUSTER {cluster_id} PREPROCESSING RESULTS")
    print(f"=" * 60)
    print(f"Total comparisons: {total_comparisons}")
    print(f"📋 Candidates for ML analysis: {len(candidates_for_ml)}")
    print(f"👥 Filtered (same authors): {len(filtered_same_author)}")
    print(f"🚫 Filtered (no recurring phrases): {len(filtered_no_phrases)}")

    if total_comparisons > 0:
        efficiency = len(filtered_no_phrases) / total_comparisons * 100
        print(f"⚡ Filtering efficiency: {efficiency:.1f}% (avoided expensive ML)")

    # Show candidates with authors
    if candidates_for_ml:
        print(f"\n📋 CANDIDATE PAIRS WITH AUTHORS:")
        print("-" * 50)
        for idx, candidate in enumerate(candidates_for_ml, 1):
            name_a = os.path.basename(candidate['file_a'])
            name_b = os.path.basename(candidate['file_b'])
            signal = candidate['signal_strength']
            priority = candidate['priority'].upper()

            print(f"{idx}. {name_a} <-> {name_b}")
            print(f"   Signal: {signal} | Priority: {priority}")

            # Display authors
            if candidate['authors_a']:
                authors_str = ", ".join(candidate['authors_a'])
                print(f"   Authors A: {authors_str}")
            else:
                print(f"   Authors A: [No authors found]")

            if candidate['authors_b']:
                authors_str = ", ".join(candidate['authors_b'])
                print(f"   Authors B: {authors_str}")
            else:
                print(f"   Authors B: [No authors found]")
            print()

    # Show high-priority candidates
    high_priority = [c for c in candidates_for_ml if c['priority'] == 'high']
    if high_priority:
        print(f"\n🔥 HIGH PRIORITY candidates (signal > 2000):")
        for candidate in high_priority[:5]:  # Show top 5
            name_a = os.path.basename(candidate['file_a'])
            name_b = os.path.basename(candidate['file_b'])
            print(f"  • {name_a} <-> {name_b} (signal: {candidate['signal_strength']})")

    return candidates_for_ml

def main():
    parser = argparse.ArgumentParser(description='Unified Bloom Filter Pipeline')
    parser.add_argument('mode', choices=['preprocess', 'test'],
                        help='Mode: preprocess clusters or test specific pair')
    parser.add_argument('--paper_a', help='First paper for testing (e.g., 1602.06707v1.tex)')
    parser.add_argument('--paper_b', help='Second paper for testing')
    parser.add_argument('--threshold', type=int, help='Override default threshold')
    parser.add_argument('--show-config', action='store_true', help='Show current configuration')

    args = parser.parse_args()

    if args.show_config:
        print("Current Configuration:")
        for key, value in BLOOM_CONFIG.items():
            print(f"  {key}: {value}")
        return

    if args.threshold:
        BLOOM_CONFIG['threshold'] = args.threshold
        print(f"Threshold overridden to: {args.threshold}")

    if args.mode == 'preprocess':
        preprocess_clusters()

    elif args.mode == 'test':
        if not args.paper_a or not args.paper_b:
            print("Error: --paper_a and --paper_b required for test mode")
            return

        # Find files in cluster data
        cluster_file = "cluster_results.json"
        if not os.path.exists(cluster_file):
            print("cluster_results.json not found!")
            return

        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)

        # Find the target files
        file_a, file_b = None, None
        for cluster_id, files in cluster_data.items():
            for file_path in files:
                if args.paper_a in file_path:
                    file_a = file_path
                elif args.paper_b in file_path:
                    file_b = file_path

        if not file_a or not file_b:
            print(f"Could not find papers: {args.paper_a} and/or {args.paper_b}")
            return

        # Run accuracy test
        actual_overlaps, bloom_overlap_count, overlapping_ngrams = test_bloom_accuracy(file_a, file_b)


if __name__ == "__main__":
    main()

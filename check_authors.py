#!/usr/bin/env python3
import re
import sys
import os
import json

def find_file(filename):
    # First try current directory
    if os.path.exists(filename):
        return filename
    
    # Try to find in cluster_results.json
    if os.path.exists('cluster_results.json'):
        with open('cluster_results.json', 'r') as f:
            cluster_data = json.load(f)
        
        for cluster_files in cluster_data.values():
            for file_path in cluster_files:
                if filename in file_path:
                    return file_path
    
    return filename  # fallback

def check_authors(file_path):
    full_path = find_file(file_path)
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"=== {file_path} ===")
    
    # Find all \author{...} patterns
    authors = re.findall(r'\\author\{([^}]+)\}', content, re.IGNORECASE)
    print(f"Raw author patterns: {authors}")
    
    # Find \author[...]{...} patterns  
    author_opts = re.findall(r'\\author\[([^\]]*)\]\{([^}]+)\}', content, re.IGNORECASE)
    print(f"Author with options: {author_opts}")
    
    # Show first few lines with 'author' in them
    lines = content.split('\n')
    author_lines = [f"Line {i+1}: {line.strip()}" for i, line in enumerate(lines) if 'author' in line.lower()][:5]
    print("Lines containing 'author':")
    for line in author_lines:
        print(f"  {line}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_authors.py file.tex")
    else:
        check_authors(sys.argv[1])

#!/usr/bin/env python3
"""
LaTeX Paper Clustering by Abstract
Extract abstracts from .tex files and cluster them using the same pipeline
"""

import os
import re
import sys
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
import matplotlib
matplotlib.use('Agg')


def extract_abstract_from_tex(tex_content):
    """Extract abstract from LaTeX content - handles many variations"""
    
    # All possible abstract patterns
    patterns = [
        # Standard environments
        r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
        r'\\begin\{Abstract\}(.*?)\\end\{Abstract\}',
        r'\\begin\{ABSTRACT\}(.*?)\\end\{ABSTRACT\}',
        r'\\begin\{abstract\*\}(.*?)\\end\{abstract\*\}',
        
        # Command-style abstracts
        r'\\abstract\{(.*?)\}',
        r'\\Abstract\{(.*?)\}',
        
        # Section-based abstracts (most common alternative)
        r'\\section\*\{abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\section\*\{Abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\section\*\{ABSTRACT\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        
        # Section without asterisk
        r'\\section\{abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\section\{Abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\section\{ABSTRACT\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        
        # Subsection-based abstracts
        r'\\subsection\*\{abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\subsection\*\{Abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\subsection\{abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\subsection\{Abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        
        # Paragraph-based abstracts
        r'\\paragraph\{abstract\}(.*?)(?=\\section|\\subsection|\\paragraph|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\paragraph\{Abstract\}(.*?)(?=\\section|\\subsection|\\paragraph|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        
        # Bold text abstracts
        r'\\textbf\{abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\textbf\{Abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        
        # Summary as alternative
        r'\\section\*\{summary\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\section\*\{Summary\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\begin\{summary\}(.*?)\\end\{summary\}',
        
        # Non-English abstracts
        r'\\section\*\{resumo\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',  # Portuguese
        r'\\section\*\{resumen\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',  # Spanish
        r'\\section\*\{zusammenfassung\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',  # German
        r'\\section\*\{résumé\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',  # French
        
        # Numbered sections (sometimes abstracts are section 1)
        r'\\section\{1\.\s*abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\section\{1\.\s*Abstract\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        
        # Just bold Abstract: at start of document
        r'\\textbf\{Abstract:\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\textbf\{ABSTRACT:\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        
        # Abstract with colon
        r'\\section\*\{abstract:\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
        r'\\section\*\{Abstract:\}(.*?)(?=\\section|\\subsection|\\chapter|\\bibliographystyle|\\bibliography|\\end\{document\}|\Z)',
    ]
    
    # Try each pattern
    for i, pattern in enumerate(patterns):
        match = re.search(pattern, tex_content, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1)
            # Clean the abstract
            abstract = clean_latex_text(abstract)
            abstract_clean = abstract.strip()
            
            # DEBUG: Print what we found for first few files
            #if len(abstract_clean) > 0:
                #print(f"    DEBUG: Pattern {i+1} matched, raw length: {len(abstract)}, clean length: {len(abstract_clean)}")
                #print(f"    DEBUG: First words: {' '.join(abstract_clean.split()[:10])}")
                #print(f"    DEBUG: Word count: {len(abstract_clean.split())}")
            
            # FIXED validation logic - made more lenient
            if (30 <= len(abstract_clean) <= 8000 and  # Reduced minimum from 50 to 30, increased max
                len(abstract_clean.split()) >= 8):      # Reduced minimum words from 10 to 8
                
                # Additional checks to filter out obvious non-abstracts
                lower_abstract = abstract_clean.lower()
                
                # Skip if it looks like a table of contents or reference
                if any(phrase in lower_abstract for phrase in [
                    'table of contents', 'contents', 'bibliography', 'references',
                    'acknowledgments', 'acknowledgements', 'appendix'
                ]):
                    #print(f"    DEBUG: Skipped - contains non-abstract content")
                    continue
                
                # Skip if it's just a single sentence that's too generic
                sentences = abstract_clean.split('.')
                if len(sentences) <= 2 and any(phrase in lower_abstract for phrase in [
                    'this document', 'this file', 'see section', 'page '
                ]):
                    #print(f"    DEBUG: Skipped - too generic/short")
                    continue
                
                #print(f"    DEBUG: Abstract accepted!")
                return abstract_clean
            #else:
                #print(f"    DEBUG: Abstract rejected - length: {len(abstract_clean)}, words: {len(abstract_clean.split())}")
    
    return None


def clean_latex_text(text):
    """Clean LaTeX commands from text - Enhanced version"""
    # Remove comments first
    text = re.sub(r'^%.*$', '', text, flags=re.MULTILINE)
    
    # Handle special LaTeX characters and commands
    replacements = {
        r'\\@': '',  # Remove \@ 
        r'\\\\': ' ',  # Replace \\ with space
        r'\\,': ' ',   # Replace \, with space
        r'\\;': ' ',   # Replace \; with space
        r'\\!': '',    # Remove \!
        r'\\&': '&',   # Replace \& with &
        r'\\_': '_',   # Replace \_ with _
        r'\\#': '#',   # Replace \# with #
        r'\\\$': '$',  # Replace \$ with $
        r'\\%': '%',   # Replace \% with %
        r'\\\{': '{',  # Replace \{ with {
        r'\\\}': '}',  # Replace \} with }
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove math environments
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', text, flags=re.DOTALL)
    
    # Remove citations and references
    text = re.sub(r'\\cite[tp]?\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{[^}]*\}', '', text)
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    
    # Remove common LaTeX commands with arguments (more comprehensive)
    # Handle nested braces better
    latex_commands = [
        r'\\[a-zA-Z]+\*?\[[^\]]*\]\{[^}]*\}',  # Commands with optional arguments
        r'\\[a-zA-Z]+\*?\{[^}]*\}',            # Simple commands with arguments
        r'\\url\{[^}]*\}',                     # URLs
        r'\\href\{[^}]*\}\{[^}]*\}',          # Hyperlinks
    ]
    
    for cmd_pattern in latex_commands:
        text = re.sub(cmd_pattern, '', text)
    
    # Remove remaining LaTeX commands (without arguments)
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
    
    # Clean up remaining braces and special characters
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'~', ' ', text)  # Replace ~ with space
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\s*\.\s*', '. ', text)  # Fix spacing around periods
    text = re.sub(r'\s*,\s*', ', ', text)   # Fix spacing around commas
    
    # Clean up whitespace
    text = text.strip()
    
    return text

def find_and_process_tex_files(base_dir, limit=None):
    """Find all .tex files and extract their abstracts"""
    print(f"Scanning for .tex files in {base_dir}...")
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory {base_dir} does not exist!")
        return []
    
    print(f"Directory exists. Checking contents...")
    
    # Show some directory structure for debugging
    try:
        top_level = os.listdir(base_dir)
        print(f"Top level directories/files: {top_level[:10]}...")  # Show first 10
    except Exception as e:
        print(f"Error listing directory: {e}")
        return []
    
    # Find all tex files
    print(f"Starting recursive search with pattern: {base_dir}/**/*.tex")
    pattern = os.path.join(base_dir, "**", "*.tex")
    
    print("This might take a while for large directory structures...")
    tex_files = []
    
    # Alternative approach: walk through directories manually with progress
    file_count = 0
    dir_count = 0
    
    for root, dirs, files in os.walk(base_dir):
        dir_count += 1
        if dir_count % 100 == 0:
            print(f"Searched {dir_count} directories, found {file_count} .tex files so far...")
        
        for file in files:
            if file.endswith('.tex'):
                tex_files.append(os.path.join(root, file))
                file_count += 1
                
                if file_count % 50 == 0:
                    print(f"Found {file_count} .tex files...")
                
                if limit and file_count >= limit:
                    print(f"Reached limit of {limit} files")
                    break
        
        if limit and file_count >= limit:
            break
    
    print(f"Found {len(tex_files)} .tex files total")
    
    # Process each file
    papers_data = []
    successful_extractions = 0
    failed_extractions = 0
    
    print("\nStarting abstract extraction...")
    
    for i, tex_file in enumerate(tex_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(tex_files)} - Success: {successful_extractions}, Failed: {failed_extractions}")
        
        # Show first few files being processed
        if i < 5:
            print(f"  Processing: {tex_file}")
        
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                tex_content = f.read()
            
            # Show file size for first few files
            if i < 5:
                print(f"    File size: {len(tex_content)} characters")
            
            # Extract abstract
            abstract = extract_abstract_from_tex(tex_content)
            
            if i < 5:
                if abstract:
                    print(f"    Abstract found: {len(abstract)} characters")
                    print(f"    Preview: {abstract[:100]}...")
                else:
                    print(f"    No abstract found")
            
            if abstract and len(abstract) > 50:  # Only keep papers with meaningful abstracts
                papers_data.append({
                    'file_path': tex_file,
                    'abstract': abstract,
                    'title': os.path.basename(tex_file).replace('.tex', '')
                })
                successful_extractions += 1
            else:
                failed_extractions += 1
        
        except Exception as e:
            if i < 10:  # Show first 10 errors
                print(f"Error processing {tex_file}: {e}")
            failed_extractions += 1
            continue
    
    print(f"\nAbstract extraction complete!")
    print(f"Successfully extracted {successful_extractions} abstracts")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Total papers with valid abstracts: {len(papers_data)}")
    return papers_data


# 1. Document Embedding (same as your reference)
def create_embeddings(abstracts):
    # Load pre-trained SBERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Generate embeddings for abstracts
    embeddings = model.encode(abstracts)
    return embeddings


# 2. Dimension Reduction (same as your reference)
def reduce_dimensions(embeddings, n_components=5):
    # Apply UMAP for dimensionality reduction
    umap_model = UMAP(n_components=n_components,
                      n_neighbors=15,
                      min_dist=0.1,
                      random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    return reduced_embeddings


# 3. Document Clustering (same as your reference)
def cluster_documents(reduced_embeddings):
    # Apply HDBSCAN for clustering
    clusterer = HDBSCAN(min_cluster_size=10,
                        min_samples=2,
                        metric='euclidean')
    clusters = clusterer.fit_predict(reduced_embeddings)
    return clusters


# 4. Get clusters and their file paths (modified from your reference)
def get_cluster_documents(df, clusters):
    df['cluster'] = clusters
    # Return a dictionary of cluster IDs mapped to file paths
    cluster_docs = {}
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        if cluster_id != -1:  # Skip noise points (HDBSCAN assigns -1 to noise)
            cluster_files = df[df['cluster'] == cluster_id]['file_path'].tolist()
            cluster_docs[cluster_id] = cluster_files
    return cluster_docs


# Main function to run the pipeline (same logic as your reference)
def semantic_prefilter(df):
    abstracts = df['abstract'].tolist()

    # Step 1: Create embeddings
    print("Creating embeddings...")
    embeddings = create_embeddings(abstracts)

    # Step 2: Reduce dimensions
    print("Reducing dimensions...")
    reduced_embeddings = reduce_dimensions(embeddings)

    # Step 3: Cluster documents
    print("Clustering documents...")
    clusters = cluster_documents(reduced_embeddings)

    # Step 4: Get clusters and their documents
    cluster_docs = get_cluster_documents(df, clusters)

    return cluster_docs, embeddings, clusters


# Visualization functions (same as your reference)
def visualize_clusters(embeddings, clusters, save_path="cluster_visualization.png"):
    # First reduce to 2D for visualization regardless of previous reduction
    umap_2d = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    vis_embeddings = umap_2d.fit_transform(embeddings)

    # Create a DataFrame for easier plotting
    vis_df = pd.DataFrame({
        'x': vis_embeddings[:, 0],
        'y': vis_embeddings[:, 1],
        'cluster': clusters
    })

    # Set up plot size
    plt.figure(figsize=(20, 16))

    # Create a color palette that handles noise points (-1) separately
    unique_clusters = np.unique(clusters)
    num_clusters = len([c for c in unique_clusters if c != -1])
    palette = sns.color_palette("hsv", num_clusters)
    colors = {i: palette[i] for i in range(num_clusters)}
    colors[-1] = (0.5, 0.5, 0.5)  # Gray for noise points

    # Plot each cluster
    for cluster_id in unique_clusters:
        cluster_data = vis_df[vis_df['cluster'] == cluster_id]
        plt.scatter(
            cluster_data['x'],
            cluster_data['y'],
            c=[colors[cluster_id]] * len(cluster_data),
            label=f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise',
            alpha=0.7 if cluster_id != -1 else 0.3,
            s=50
        )

    # Add labels and title
    plt.title('Document Clusters Visualization', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)

    # Place the legend at the bottom of the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=min(13, len(unique_clusters)),
               frameon=True, fancybox=True, shadow=True)

    # Adjust layout to make room for the legend at the bottom
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    # Save the plot instead of showing it
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Cluster visualization saved to: {save_path}")

    return save_path


def visualize_cluster_sample(df, embeddings, clusters, n_samples=3, save_path="cluster_visualization.png"):
    """Visualize clusters and display sample titles from each cluster"""
    plot_path = visualize_clusters(embeddings, clusters, save_path)

    # Display sample documents from each cluster
    print("Sample documents from each cluster:")
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue  # Skip noise points

        cluster_docs = df[df['cluster'] == cluster_id]
        sample_docs = cluster_docs.sample(min(n_samples, len(cluster_docs)))

        print(f"\nCluster {cluster_id} ({len(cluster_docs)} documents):")
        for idx, row in sample_docs.iterrows():
            print(f"  - {row['title']}")

    return plot_path


# Download necessary NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass


def get_top_words_per_cluster(df, clusters, n_words=3):
    """Get the most common words for each cluster"""
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()

    # Add domain-specific stopwords
    domain_stops = {'using', 'paper', 'approach', 'method', 'model', 'propose', 'based', 'results', 'data', 'proposed'}
    stop_words.update(domain_stops)

    cluster_top_words = {}
    unique_clusters = np.unique(clusters)

    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise points
            continue

        # Get all abstracts for this cluster
        cluster_docs = df[clusters == cluster_id]

        # Combine all text
        all_text = ' '.join(cluster_docs['abstract'].fillna(''))

        # Basic preprocessing
        all_text = all_text.lower()
        # Remove special characters and numbers
        all_text = re.sub(r'[^a-zA-Z\s]', '', all_text)

        # Tokenize
        words = all_text.split()

        # Remove stopwords
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]

        # Get most common words
        word_counts = Counter(filtered_words)
        top_words = [word for word, count in word_counts.most_common(n_words)]

        cluster_top_words[cluster_id] = top_words

    return cluster_top_words


def main():
    # Configuration
    if len(sys.argv) < 2:
        base_dir = "/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/2024_25/arxiv_data/full_papers"
        print(f"Using default directory: {base_dir}")
    else:
        base_dir = sys.argv[1]
    
    # For testing, limit the number of papers
    max_papers = 20000  # Reduced for initial testing
    print(f"Will process maximum {max_papers} papers")
    
    print("=" * 60)
    print("LaTeX Paper Clustering by Abstract")
    print("=" * 60)
    
    # Step 1: Find and process tex files
    papers_data = find_and_process_tex_files(base_dir, limit=max_papers)
    
    if len(papers_data) < 10:
        print(f"Not enough papers with abstracts found ({len(papers_data)}). Exiting.")
        return
    
    # Step 2: Create DataFrame
    df = pd.DataFrame(papers_data)
    print(f"Created DataFrame with {len(df)} papers")
    
    # Step 3: Run the clustering pipeline
    cluster_docs, embeddings, clusters = semantic_prefilter(df)
    
    # Step 4: Print results
    print(f"\nClustering Results:")
    print(f"Number of clusters found: {len(cluster_docs)}")
    print(f"Distribution of documents across clusters: {[len(docs) for docs in cluster_docs.values()]}")
    
    # Step 5: Show cluster statistics
    for cluster_id, file_paths in cluster_docs.items():
        print(f"\nCluster {cluster_id}: {len(file_paths)} papers")
        # Show first few file names
        for i, file_path in enumerate(file_paths[:3]):
            print(f"  - {os.path.basename(file_path)}")
        if len(file_paths) > 3:
            print(f"  ... and {len(file_paths) - 3} more")
    
    # Step 6: Visualize (optional)
    try:
        visualize_cluster_sample(df, embeddings, clusters)
        print("Cluster visualization saved 'cluster_visualization.png'")

    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Step 7: Save results for later use
    output_file = "cluster_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        cluster_data = {
            str(cluster_id): file_paths 
            for cluster_id, file_paths in cluster_docs.items()
        }
        json.dump(cluster_data, f, indent=2)
    
    print(f"\nCluster results saved to {output_file}")
    print("Ready for bloom filter processing within each cluster!")


if __name__ == "__main__":
    main()

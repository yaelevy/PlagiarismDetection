#!/usr/bin/env python3
"""
Plagiarism Detection Inference Pipeline
Applies trained Siamese BERT model to Bloom filter candidates
Usage: python inference_pipeline.py --model_path best_siamese_bert.pth --candidates_file bloom_candidates.json [options]
"""

import argparse
import json
import os
import re
import torch
import pandas as pd
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
import time
from tqdm import tqdm

# Import the model class from the training script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from plagiarism_detector import SiameseBERT, predict_similarity


def extract_content_for_inference(tex_file_path: str) -> str:
    """
    Extract clean text content from LaTeX file for plagiarism detection
    Uses same logic as training preprocessing for consistency
    """
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

        # Remove non-textual environments
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


        return tex_content.strip()

    except Exception as e:
        print(f"Error reading {tex_file_path}: {e}")
        return ""


def chunk_text_for_bert(text: str, tokenizer, max_length: int = 512) -> List[str]:
    """
    Split long text into chunks that fit BERT's token limit
    Returns list of text chunks
    """
    if not text:
        return []
    
    # Quick check if text fits in one chunk
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) <= max_length:
        return [text]
    
    # Split into sentences first
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence exceeds limit
        test_chunk = f"{current_chunk} {sentence}".strip()
        test_tokens = tokenizer.encode(test_chunk, add_special_tokens=True)
        
        if len(test_tokens) <= max_length:
            current_chunk = test_chunk
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
            # Start new chunk with current sentence
            current_chunk = sentence
            
            # If single sentence is still too long, truncate it
            single_tokens = tokenizer.encode(sentence, add_special_tokens=True)
            if len(single_tokens) > max_length:
                # Decode truncated tokens back to text
                truncated_tokens = single_tokens[:max_length-1]  # Leave room for SEP token
                current_chunk = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)



    return chunks if chunks else [text[:1000]]  # Fallback


def compare_papers_with_chunking(model: SiameseBERT, tokenizer, file_a: str, file_b: str, 
                                threshold: float = 0.5, device: str = 'cuda') -> Dict:
    """
    Compare two papers using chunking strategy for long documents
    Returns aggregated similarity result
    """
    # Extract text content
    text_a = extract_content_for_inference(file_a)
    text_b = extract_content_for_inference(file_b)
    
    if not text_a or not text_b:
        return {
            'similarity_score': 0.0,
            'is_plagiarized': False,
            'error': 'Could not extract text from one or both files',
            'chunks_compared': 0
        }
    
    # Create chunks
    chunks_a = chunk_text_for_bert(text_a, tokenizer)
    chunks_b = chunk_text_for_bert(text_b, tokenizer)
    
    print(f"  Text A: {len(text_a)} chars → {len(chunks_a)} chunks")
    print(f"  Text B: {len(text_b)} chars → {len(chunks_b)} chunks")
    
    # Compare all chunk pairs and collect similarity scores
    similarities = []
    max_similarity = 0.0
    
    for chunk_a in chunks_a:
        for chunk_b in chunks_b:
            print("chunk_a:", chunk_a, "chunk_b:", chunk_b)
            result = predict_similarity(model, tokenizer, chunk_a, chunk_b, 
                                      threshold=None, device=device)
            score = result['similarity_score']
            similarities.append(score)
            max_similarity = max(max_similarity, score)
    
    if not similarities:
        return {
            'similarity_score': 0.0,
            'is_plagiarized': False,
            'error': 'No valid chunks to compare',
            'chunks_compared': 0
        }
    
    # Aggregation strategies
    mean_similarity = sum(similarities) / len(similarities)
    median_similarity = sorted(similarities)[len(similarities) // 2]
    top_10_percent = sorted(similarities, reverse=True)[:max(1, len(similarities) // 10)]
    top_10_mean = sum(top_10_percent) / len(top_10_percent)
    
    # Use top-10% mean as final score (focuses on most similar parts)
    final_score = top_10_mean
    
    return {
        'similarity_score': final_score,
        'max_similarity': max_similarity,
        'mean_similarity': mean_similarity,
        'median_similarity': median_similarity,
        'is_plagiarized': final_score > threshold,
        'chunks_compared': len(similarities),
        'chunks_a': len(chunks_a),
        'chunks_b': len(chunks_b),
        'threshold_used': threshold
    }


def run_inference_on_candidates(model_path: str, candidates_file: str, 
                               threshold: float = 0.5, output_file: str = None,
                               max_candidates: int = None, 
                               prioritize_high: bool = True) -> List[Dict]:
    """
    Run Siamese BERT inference on Bloom filter candidates
    """
    print("=" * 80)
    print("PLAGIARISM DETECTION INFERENCE PIPELINE")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Candidates: {candidates_file}")
    print(f"Threshold: {threshold}")
    print(f"Max candidates: {max_candidates or 'All'}")
    print("=" * 80)
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(candidates_file):
        raise FileNotFoundError(f"Candidates file not found: {candidates_file}")
    
    # Load candidates
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)
    
    print(f"Loaded {len(candidates)} candidate pairs from Bloom filter")
    
    # Filter and sort candidates
    if prioritize_high:
        # Sort by priority (high first) then by signal strength
        candidates.sort(key=lambda x: (
            0 if x.get('priority') == 'high' else 1,
            -x.get('signal_strength', 0)
        ))
    else:
        # Sort by signal strength only
        candidates.sort(key=lambda x: -x.get('signal_strength', 0))
    
    # Limit candidates if specified
    if max_candidates:
        candidates = candidates[:max_candidates]
        print(f"Limited to top {max_candidates} candidates")
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = SiameseBERT('bert-base-uncased')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print()
    
    # Process candidates
    results = []
    start_time = time.time()
    
    for i, candidate in enumerate(tqdm(candidates, desc="Processing candidates")):
        file_a = candidate['file_a']
        file_b = candidate['file_b']
        
        print(f"\n[{i+1}/{len(candidates)}] Processing:")
        print(f"  A: {os.path.basename(file_a)}")
        print(f"  B: {os.path.basename(file_b)}")
        print(f"  Bloom signal: {candidate.get('signal_strength', 'N/A')}")
        print(f"  Priority: {candidate.get('priority', 'N/A')}")
        
        # Check if files exist
        if not os.path.exists(file_a) or not os.path.exists(file_b):
            print(f"  ❌ SKIPPED: One or both files not found")
            continue
        
        # Run inference
        try:
            result = compare_papers_with_chunking(
                model, tokenizer, file_a, file_b, threshold, device
            )
            
            # Add metadata
            result.update({
                'file_a': file_a,
                'file_b': file_b,
                'file_a_name': os.path.basename(file_a),
                'file_b_name': os.path.basename(file_b),
                'bloom_signal_strength': candidate.get('signal_strength'),
                'bloom_priority': candidate.get('priority'),
                'cluster_id': candidate.get('cluster_id'),
                'authors_a': candidate.get('authors_a', []),
                'authors_b': candidate.get('authors_b', []),
                'processing_order': i + 1
            })
            
            results.append(result)
            
            # Display result
            score = result['similarity_score']
            is_plagiarized = result['is_plagiarized']
            chunks_compared = result['chunks_compared']
            
            status = "🚨 PLAGIARISM DETECTED" if is_plagiarized else "✅ No plagiarism"
            print(f"  {status}")
            print(f"  Final similarity: {score:.4f}")
            print(f"  Max chunk similarity: {result.get('max_similarity', 0):.4f}")
            print(f"  Chunks compared: {chunks_compared}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append({
                'file_a': file_a,
                'file_b': file_b,
                'file_a_name': os.path.basename(file_a),
                'file_b_name': os.path.basename(file_b),
                'similarity_score': 0.0,
                'is_plagiarized': False,
                'error': str(e),
                'bloom_signal_strength': candidate.get('signal_strength'),
                'bloom_priority': candidate.get('priority'),
                'cluster_id': candidate.get('cluster_id'),
                'processing_order': i + 1
            })
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Final analysis
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"📊 Total pairs analyzed: {len(results)}")
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    error_results = [r for r in results if 'error' in r]
    
    print(f"✅ Successful analyses: {len(successful_results)}")
    print(f"❌ Errors: {len(error_results)}")
    
    if successful_results:
        plagiarism_detected = [r for r in successful_results if r['is_plagiarized']]
        print(f"🚨 Plagiarism detected: {len(plagiarism_detected)}")
        
        # Top plagiarism candidates
        if plagiarism_detected:
            plagiarism_detected.sort(key=lambda x: x['similarity_score'], reverse=True)
            print(f"\n🏆 TOP PLAGIARISM CASES:")
            for i, result in enumerate(plagiarism_detected[:5]):
                score = result['similarity_score']
                bloom_signal = result.get('bloom_signal_strength', 'N/A')
                print(f"  {i+1}. {result['file_a_name']} <-> {result['file_b_name']}")
                print(f"     BERT similarity: {score:.4f} | Bloom signal: {bloom_signal}")
        
        # Statistics
        scores = [r['similarity_score'] for r in successful_results]
        print(f"\n📈 SIMILARITY STATISTICS:")
        print(f"  Mean: {sum(scores)/len(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
    
    # Save results
    if output_file is None:
        output_file = f"inference_results_{int(time.time())}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Create summary CSV
    csv_file = output_file.replace('.json', '.csv')
    df_data = []
    for result in results:
        df_data.append({
            'file_a': result['file_a_name'],
            'file_b': result['file_b_name'],
            'similarity_score': result.get('similarity_score', 0),
            'is_plagiarized': result.get('is_plagiarized', False),
            'bloom_signal': result.get('bloom_signal_strength', 0),
            'bloom_priority': result.get('bloom_priority', ''),
            'cluster_id': result.get('cluster_id', ''),
            'chunks_compared': result.get('chunks_compared', 0),
            'error': result.get('error', '')
        })
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_file, index=False)
    print(f"📊 Summary saved to: {csv_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Plagiarism Detection Inference Pipeline')
    parser.add_argument('--model_path', required=True, help='Path to trained Siamese BERT model')
    parser.add_argument('--candidates_file', default='bloom_candidates.json', 
                        help='Path to Bloom filter candidates JSON file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity threshold for plagiarism detection')
    parser.add_argument('--output_file', help='Output file for results (default: auto-generated)')
    parser.add_argument('--max_candidates', type=int, 
                        help='Maximum number of candidates to process')
    parser.add_argument('--prioritize_high', action='store_true', default=True,
                        help='Prioritize high-priority candidates first')
    parser.add_argument('--test_single', nargs=2, metavar=('PAPER_A', 'PAPER_B'),
                        help='Test single pair of papers')
    
    args = parser.parse_args()
    
    if args.test_single:
        # Single pair testing mode
        paper_a, paper_b = args.test_single
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = SiameseBERT('bert-base-uncased')
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        
        print(f"Testing similarity between:")
        print(f"  A: {paper_a}")
        print(f"  B: {paper_b}")
        print()
        
        result = compare_papers_with_chunking(
            model, tokenizer, paper_a, paper_b, args.threshold, device
        )
        
        print("Results:")
        print(f"  Similarity score: {result['similarity_score']:.4f}")
        print(f"  Is plagiarized: {result['is_plagiarized']}")
        print(f"  Chunks compared: {result['chunks_compared']}")
        
    else:
        # Full pipeline mode
        results = run_inference_on_candidates(
            model_path=args.model_path,
            candidates_file=args.candidates_file,
            threshold=args.threshold,
            output_file=args.output_file,
            max_candidates=args.max_candidates,
            prioritize_high=args.prioritize_high
        )


if __name__ == "__main__":
    main()

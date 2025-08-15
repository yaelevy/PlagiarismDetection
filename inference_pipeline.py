#!/usr/bin/env python3
"""
Plagiarism Detection Inference Pipeline
Applies trained Siamese BERT model to Bloom filter candidates
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
from plagiarism_detector import SiameseBERT


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
        tex_content = re.sub(r'\n\s*\n', ' PARAGRAPH_BREAK ', tex_content)

        # Remove LaTeX formatting but preserve content
        tex_content = re.sub(r'\\textbf\{([^}]*)\}', r'\1', tex_content)
        tex_content = re.sub(r'\\textit\{([^}]*)\}', r'\1', tex_content)
        tex_content = re.sub(r'\\emph\{([^}]*)\}', r'\1', tex_content)
        tex_content = re.sub(r'\\text\{([^}]*)\}', r'\1', tex_content)

        # Remove other LaTeX commands
        tex_content = re.sub(r'\\[a-zA-Z]+\*?\[[^\]]*\]\{[^}]*\}', '', tex_content)
        tex_content = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', tex_content)
        tex_content = re.sub(r'\\[a-zA-Z]+\*?', '', tex_content)

        # Remove section markers that are just numbers/symbols on their own line
        tex_content = re.sub(r'\n\s*-?\d+\s*\n', '\n\n', tex_content)

        # Clean up
        tex_content = re.sub(r'[{}]', '', tex_content)
        tex_content = re.sub(r'~', ' ', tex_content)
        # Preserve paragraph breaks but clean up other whitespace
        tex_content = re.sub(r'[ \t]+', ' ', tex_content)  # Only collapse spaces and tabs
        tex_content = re.sub(r'\n[ \t]*\n', '\n\n', tex_content)  # Normalize paragraph breaks
        tex_content = re.sub(r'\n{3,}', '\n\n', tex_content)  # Limit to double newlines max

        # Restore paragraph breaks (handle extra spaces around the placeholder)
        tex_content = re.sub(r'\s*PARAGRAPH_BREAK\s*', '\n\n', tex_content)


        return tex_content.strip()

    except Exception as e:
        print(f"Error reading {tex_file_path}: {e}")
        return ""


def chunk_text_for_bert(text: str, tokenizer, max_length: int = 512) -> List[str]:
    """
    Split long text into chunks that fit BERT's token limit, prioritizing paragraph boundaries.
    Forces breaks at section boundaries (standalone numbers/markers).
    If a paragraph exceeds the token limit, split it at sentence boundaries.
    Returns list of text chunks
    """
    if not text:
        return []

    # Quick check if text fits in one chunk
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) <= max_length:
        return [text]

    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text.strip())

    # Identify section breaks (standalone numbers/markers OR short title-like paragraphs)
    section_breaks = set()
    for i, para in enumerate(paragraphs):
        para_clean = para.strip()
        # Standalone numbers (sections)
        if re.match(r'^\s*-?\d+\s*$', para_clean):
            section_breaks.add(i)
        # Short title-like paragraphs (subsections) - typically < 80 chars, no periods
        elif len(para_clean) < 80 and not para_clean.endswith('.') and len(para_clean.split()) <= 8:
            section_breaks.add(i)

    chunks = []
    current_chunk = ""

    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Force chunk break at section boundaries
        if i in section_breaks:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            continue  # Skip the section marker itself

        # Check if current chunk + this paragraph fits
        test_chunk = f"{current_chunk}\n\n{paragraph}".strip() if current_chunk else paragraph
        test_tokens = tokenizer.encode(test_chunk, add_special_tokens=True)

        if len(test_tokens) <= max_length and not current_chunk:  # Only combine if starting fresh
            current_chunk = test_chunk
        else:
            # Force paragraph boundary
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph if len(tokenizer.encode(paragraph, add_special_tokens=True)) <= max_length else ""

            # Check if paragraph itself exceeds limit
            para_tokens = tokenizer.encode(paragraph, add_special_tokens=True)
            if len(para_tokens) <= max_length:
                current_chunk = paragraph
            else:
                # Split oversized paragraph by sentences
                para_chunks = split_paragraph_by_sentences(paragraph, tokenizer, max_length)
                # Add all but the last chunk
                chunks.extend(para_chunks[:-1])
                # Set the last chunk as current
                current_chunk = para_chunks[-1] if para_chunks else ""

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:1000]]  # Fallback


def split_paragraph_by_sentences(paragraph: str, tokenizer, max_length: int) -> List[str]:
    """
    Split a paragraph that exceeds token limit by sentence boundaries
    """
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
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

            # Handle oversized single sentence
            single_tokens = tokenizer.encode(sentence, add_special_tokens=True)
            if len(single_tokens) > max_length:
                # Truncate sentence to fit
                truncated_tokens = single_tokens[:max_length - 1]  # Leave room for SEP token
                current_chunk = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            else:
                current_chunk = sentence

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def predict_similarities_batch(model: SiameseBERT, tokenizer, text_pairs: List[Tuple[str, str]],
                               threshold: float = None, device: str = 'cuda',
                               batch_size: int = 512) -> List[Dict]:
    """Predict similarities for multiple text pairs in batches"""
    model.eval()

    if not text_pairs:
        return []

    results = []
    total_batches = (len(text_pairs) + batch_size - 1) // batch_size  # Ceiling division

    # Process in batches
    for batch_idx, i in enumerate(range(0, len(text_pairs), batch_size)):
        batch_pairs = text_pairs[i:i + batch_size]

        # Print progress every few batches
        if batch_idx % 10 == 0:
            pairs_processed = min(i + batch_size, len(text_pairs))
            print(f"  Batch {batch_idx + 1}/{total_batches}: Processed {pairs_processed}/{len(text_pairs)} pairs")

        texts_1 = [pair[0] for pair in batch_pairs]
        texts_2 = [pair[1] for pair in batch_pairs]

        # Batch tokenize
        encoding1 = tokenizer(texts_1, truncation=True, padding='max_length',
                              max_length=512, return_tensors='pt')
        encoding2 = tokenizer(texts_2, truncation=True, padding='max_length',
                              max_length=512, return_tensors='pt')

        # Move to device
        input_ids_1 = encoding1['input_ids'].to(device)
        attention_mask_1 = encoding1['attention_mask'].to(device)
        input_ids_2 = encoding2['input_ids'].to(device)
        attention_mask_2 = encoding2['attention_mask'].to(device)

        # Batch inference
        with torch.no_grad():
            similarities = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)

        # Build results
        similarities_cpu = similarities.cpu().numpy()
        for j, similarity_score in enumerate(similarities_cpu):
            result = {'similarity_score': float(similarity_score)}
            if threshold is not None:
                result.update({
                    'is_plagiarized': similarity_score > threshold,
                    'threshold_used': threshold
                })
            results.append(result)

    return results


def compare_papers_with_chunking(model: SiameseBERT, tokenizer, file_a: str, file_b: str,
                                 threshold: float = 0.5, device: str = 'cuda',
                                 max_pages: int = 50) -> Dict:
    """
    Compare two papers using chunking strategy for long documents
    Returns individual chunk pairs that exceeded the threshold
    """
    # Extract text content
    text_a = extract_content_for_inference(file_a)
    text_b = extract_content_for_inference(file_b)

    if not text_a or not text_b:
        return {
            'similarity_score': 0.0,
            'is_plagiarized': False,
            'error': 'Could not extract text from one or both files',
            'chunks_compared': 0,
            'exceeding_pairs': []
        }

    # PAPER FILTER:
    # Rough estimation: ~2000 characters per page
    MAX_CHARS = max_pages * 2000

    if len(text_a) > MAX_CHARS or len(text_b) > MAX_CHARS:
        return {
            'similarity_score': 0.0,
            'is_plagiarized': False,
            'error': f'Paper too long (>{max_pages} pages estimated). File A: {len(text_a)} chars, File B: {len(text_b)} chars',
            'chunks_compared': 0,
            'skipped_reason': 'too_long',
            'exceeding_pairs': []
        }

    # Create chunks
    chunks_a = chunk_text_for_bert(text_a, tokenizer)
    chunks_b = chunk_text_for_bert(text_b, tokenizer)

    print(f"  Text A: {len(text_a)} chars → {len(chunks_a)} chunks")
    print(f"  Text B: {len(text_b)} chars → {len(chunks_b)} chunks")

    # Compare all chunk pairs and collect similarity scores
    # Create all chunk pairs with metadata
    chunk_pairs = []
    chunk_metadata = []
    for i, chunk_a in enumerate(chunks_a):
        for j, chunk_b in enumerate(chunks_b):
            chunk_pairs.append((chunk_a, chunk_b))
            chunk_metadata.append({
                'chunk_a_index': i,
                'chunk_b_index': j,
                'chunk_a_text': chunk_a,
                'chunk_b_text': chunk_b
            })

    # Batch process all pairs
    print(f"  Processing {len(chunk_pairs)} pairs in batches...")
    results = predict_similarities_batch(
        model, tokenizer, chunk_pairs, threshold=None, device=device, batch_size=512
    )

    # Extract similarity scores and find exceeding pairs
    similarities = [r['similarity_score'] for r in results]
    exceeding_pairs = []

    for i, result in enumerate(results):
        similarity_score = result['similarity_score']
        if similarity_score > threshold:
            exceeding_pair = {
                'chunk_a_index': chunk_metadata[i]['chunk_a_index'],
                'chunk_b_index': chunk_metadata[i]['chunk_b_index'],
                'chunk_a_text': chunk_metadata[i]['chunk_a_text'],
                'chunk_b_text': chunk_metadata[i]['chunk_b_text'],
                'similarity_score': similarity_score,
                'exceeded_threshold': True
            }
            exceeding_pairs.append(exceeding_pair)

    if not similarities:
        return {
            'similarity_score': 0.0,
            'is_plagiarized': False,
            'error': 'No valid chunks to compare',
            'chunks_compared': 0,
            'exceeding_pairs': []
        }

    # Keep some aggregated stats for reference
    mean_similarity = sum(similarities) / len(similarities)
    median_similarity = sorted(similarities)[len(similarities) // 2]
    top_10_percent = sorted(similarities, reverse=True)[:max(1, len(similarities) // 10)]
    top_10_mean = sum(top_10_percent) / len(top_10_percent)
    final_score = top_10_mean

    return {
        'similarity_score': final_score,
        'mean_similarity': mean_similarity,
        'median_similarity': median_similarity,
        'is_plagiarized': len(exceeding_pairs) > 0,  # True if any pairs exceeded threshold
        'chunks_compared': len(similarities),
        'chunks_a': len(chunks_a),
        'chunks_b': len(chunks_b),
        'threshold_used': threshold,
        'exceeding_pairs': exceeding_pairs,  # NEW: List of chunk pairs that exceeded threshold
        'num_exceeding_pairs': len(exceeding_pairs)  # NEW: Count of exceeding pairs
    }


def run_inference_on_candidates(model_path: str, candidates_file: str,
                                threshold: float = 0.5, output_file: str = None,
                                max_candidates: int = None,
                                prioritize_high: bool = True,
                                max_pages: int = 50) -> List[Dict]:
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
    for c in candidates:
        if c["actual_overlaps"] == 0:
            # remove the candidate
            candidates.remove(c)

    # sort the candidates by actual_overlaps
    candidates.sort(key=lambda x: x["actual_overlaps"], reverse=True)

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

        print(f"\n[{i + 1}/{len(candidates)}] Processing:")
        print(f"  A: {os.path.basename(file_a)}")
        print(f"  B: {os.path.basename(file_b)}")
        print(f" actual overlaps: {candidate['actual_overlaps']}")
        print(f"  Bloom signal: {candidate.get('signal_strength', 'N/A')}")

        # Check if files exist
        if not os.path.exists(file_a) or not os.path.exists(file_b):
            print(f"  ❌ SKIPPED: One or both files not found")
            continue

        # Run inference
        try:
            result = compare_papers_with_chunking(
                model, tokenizer, file_a, file_b, threshold, device, max_pages)

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
            exceeding_pairs = result.get('exceeding_pairs', [])

            status = "🚨 PLAGIARISM DETECTED" if is_plagiarized else "✅ No plagiarism"
            print(f"  {status}")
            print(f"  Final similarity: {score:.4f}")
            print(f"  Chunks compared: {chunks_compared}")
            print(f"  Pairs exceeding threshold: {len(exceeding_pairs)}")

            # Display the exceeding pairs
            if exceeding_pairs:
                print(f"  📋 EXCEEDING PAIRS:")
                for idx, pair in enumerate(exceeding_pairs[:3]):  # Show first 3 pairs
                    print(f"    Pair {idx + 1}: Score {pair['similarity_score']:.4f}")
                    print(f"      Chunk A[{pair['chunk_a_index']}]: {pair['chunk_a_text'][:100]}...")
                    print(f"      Chunk B[{pair['chunk_b_index']}]: {pair['chunk_b_text'][:100]}...")
                    print()
                if len(exceeding_pairs) > 3:
                    print(f"    ... and {len(exceeding_pairs) - 3} more pairs")

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
                'processing_order': i + 1,
                'exceeding_pairs': []
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
                num_exceeding = len(result.get('exceeding_pairs', []))
                print(f"  {i + 1}. {result['file_a_name']} <-> {result['file_b_name']}")
                print(
                    f"     BERT similarity: {score:.4f} | Bloom signal: {bloom_signal} | Exceeding pairs: {num_exceeding}")

        # Statistics
        scores = [r['similarity_score'] for r in successful_results]
        total_exceeding_pairs = sum(len(r.get('exceeding_pairs', [])) for r in successful_results)
        print(f"\n📈 SIMILARITY STATISTICS:")
        print(f"  Mean: {sum(scores) / len(scores):.4f}")
        print(f"  Max: {max(scores):.4f}")
        print(f"  Min: {min(scores):.4f}")
        print(f"  Total exceeding chunk pairs: {total_exceeding_pairs}")

    # Save results
    if output_file is None:
        output_file = f"inference_results_{int(time.time())}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

    return results

def main():
    parser = argparse.ArgumentParser(description='Plagiarism Detection Inference Pipeline')
    parser.add_argument('--model_path', required=True, help='Path to trained Siamese BERT model')
    parser.add_argument('--candidates_file', default='bloom_overlap_results.json',
                        help='Path to Bloom filter candidates JSON file')
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help='Similarity threshold for plagiarism detection')
    parser.add_argument('--output_file', help='Output file for results (default: auto-generated)')
    parser.add_argument('--max_candidates', type=int, 
                        help='Maximum number of candidates to process')
    parser.add_argument('--max_pages', type=int, default=50,
                        help='Skip papers longer than N pages (default: 50)')
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
            prioritize_high=args.prioritize_high,
            max_pages=args.max_pages
        )


if __name__ == "__main__":
    main()

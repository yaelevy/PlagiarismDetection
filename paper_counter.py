#!/usr/bin/env python3
"""
Count LaTeX Papers in Directory Structure
Analyzes the directory structure and counts .tex files
"""

import os
import sys
import glob
from collections import defaultdict, Counter
import time

def count_papers_by_year(base_dir):
    """Count papers organized by year and provide detailed statistics"""
    
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory {base_dir} does not exist!")
        return
    
    print(f"Scanning directory structure: {base_dir}")
    print("=" * 80)
    
    # Get top-level directories
    try:
        top_level = sorted([d for d in os.listdir(base_dir) 
                           if os.path.isdir(os.path.join(base_dir, d))])
        print(f"Top-level directories found: {len(top_level)}")
        print(f"Directories: {top_level}")
        print()
    except Exception as e:
        print(f"Error reading top-level directories: {e}")
        return
    
    # Statistics containers
    year_counts = defaultdict(int)
    month_counts = defaultdict(int)
    total_files = 0
    total_dirs = 0
    year_month_counts = defaultdict(lambda: defaultdict(int))
    
    # Track file sizes and types
    file_sizes = []
    file_extensions = Counter()
    
    start_time = time.time()
    
    print("Detailed analysis by year:")
    print("-" * 60)
    
    # Analyze each year directory
    for year_dir in top_level:
        year_path = os.path.join(base_dir, year_dir)
        
        # Skip if not a directory or not year-like
        if not os.path.isdir(year_path):
            continue
            
        # Try to parse as year
        try:
            year = int(year_dir) if year_dir.isdigit() else year_dir
        except:
            year = year_dir
        
        year_file_count = 0
        year_dir_count = 0
        
        print(f"\nAnalyzing {year_dir}...")
        
        # Walk through year directory
        for root, dirs, files in os.walk(year_path):
            year_dir_count += len(dirs)
            
            # Count .tex files
            tex_files = [f for f in files if f.endswith('.tex')]
            year_file_count += len(tex_files)
            
            # Analyze file sizes for .tex files
            for tex_file in tex_files:
                try:
                    file_path = os.path.join(root, tex_file)
                    file_size = os.path.getsize(file_path)
                    file_sizes.append(file_size)
                except:
                    pass
            
            # Count all file extensions in this directory
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext:
                    file_extensions[ext] += 1
            
            # Try to extract month information from path
            path_parts = root.replace(year_path, '').strip('/').split('/')
            if len(path_parts) >= 1 and path_parts[0].isdigit():
                month = int(path_parts[0])
                if 1 <= month <= 12:
                    month_counts[month] += len(tex_files)
                    year_month_counts[year][month] += len(tex_files)
        
        year_counts[year] = year_file_count
        total_files += year_file_count
        total_dirs += year_dir_count
        
        print(f"  {year_dir}: {year_file_count:,} .tex files in {year_dir_count:,} directories")
    
    elapsed_time = time.time() - start_time
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"Total .tex files found: {total_files:,}")
    print(f"Total directories scanned: {total_dirs:,}")
    print(f"Scan completed in: {elapsed_time:.2f} seconds")
    print(f"Average files per year: {total_files / len(year_counts):.1f}" if year_counts else "N/A")
    
    # Year breakdown
    print(f"\nBreakdown by year:")
    print("-" * 40)
    for year in sorted(year_counts.keys()):
        count = year_counts[year]
        percentage = (count / total_files) * 100 if total_files > 0 else 0
        print(f"  {year}: {count:,} files ({percentage:.1f}%)")
    
    # Month breakdown (if available)
    if month_counts:
        print(f"\nBreakdown by month (across all years):")
        print("-" * 40)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month_num in range(1, 13):
            if month_num in month_counts:
                count = month_counts[month_num]
                percentage = (count / total_files) * 100 if total_files > 0 else 0
                print(f"  {months[month_num-1]}: {count:,} files ({percentage:.1f}%)")
    
    # File size statistics
    if file_sizes:
        print(f"\nFile size statistics:")
        print("-" * 40)
        file_sizes.sort()
        print(f"  Smallest file: {min(file_sizes):,} bytes ({min(file_sizes)/1024:.1f} KB)")
        print(f"  Largest file: {max(file_sizes):,} bytes ({max(file_sizes)/1024:.1f} KB)")
        print(f"  Average file size: {sum(file_sizes)/len(file_sizes):,.0f} bytes ({sum(file_sizes)/len(file_sizes)/1024:.1f} KB)")
        print(f"  Median file size: {file_sizes[len(file_sizes)//2]:,} bytes ({file_sizes[len(file_sizes)//2]/1024:.1f} KB)")
    
    # File extension breakdown
    if file_extensions:
        print(f"\nAll file types found (top 10):")
        print("-" * 40)
        for ext, count in file_extensions.most_common(10):
            percentage = (count / sum(file_extensions.values())) * 100
            print(f"  {ext}: {count:,} files ({percentage:.1f}%)")
    
    # Yearly trends (if we have multiple years)
    years_with_data = [y for y in year_counts.keys() if isinstance(y, int)]
    if len(years_with_data) > 1:
        years_with_data.sort()
        print(f"\nYearly trends:")
        print("-" * 40)
        for i in range(1, len(years_with_data)):
            prev_year = years_with_data[i-1]
            curr_year = years_with_data[i]
            prev_count = year_counts[prev_year]
            curr_count = year_counts[curr_year]
            
            if prev_count > 0:
                growth = ((curr_count - prev_count) / prev_count) * 100
                print(f"  {prev_year} → {curr_year}: {growth:+.1f}% ({prev_count:,} → {curr_count:,})")
    
    # Estimation for clustering
    print(f"\n" + "=" * 80)
    print("CLUSTERING CONSIDERATIONS")
    print("=" * 80)
    
    if total_files > 0:
        # Estimate processing time
        files_per_minute = 60  # Rough estimate based on your output
        estimated_minutes = total_files / files_per_minute
        
        print(f"Processing estimates for full dataset:")
        print(f"  At ~{files_per_minute} files/minute: {estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)")
        
        # Memory estimates
        avg_abstract_length = 500  # characters
        embedding_size = 384  # dimensions for all-MiniLM-L6-v2
        
        memory_mb = (total_files * embedding_size * 4) / (1024 * 1024)  # 4 bytes per float
        print(f"  Estimated memory for embeddings: {memory_mb:.1f} MB")
        
        # Recommended batch sizes
        if total_files < 1000:
            print(f"  Recommended approach: Process all files at once")
        elif total_files < 10000:
            print(f"  Recommended approach: Process in batches of 1000-2000")
        else:
            print(f"  Recommended approach: Process by year or in batches of 5000")
    
    return {
        'total_files': total_files,
        'year_counts': dict(year_counts),
        'month_counts': dict(month_counts),
        'file_sizes': file_sizes,
        'file_extensions': dict(file_extensions.most_common()),
        'total_dirs': total_dirs
    }

def quick_count(base_dir):
    """Quick count without detailed analysis"""
    print(f"Quick counting .tex files in {base_dir}...")
    
    count = 0
    start_time = time.time()
    
    for root, dirs, files in os.walk(base_dir):
        tex_files = sum(1 for f in files if f.endswith('.tex'))
        count += tex_files
        
        if count % 1000 == 0 and count > 0:
            elapsed = time.time() - start_time
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  Found {count:,} files so far... ({rate:.0f} files/sec)")
    
    elapsed_time = time.time() - start_time
    print(f"\nQuick count complete: {count:,} .tex files found in {elapsed_time:.2f} seconds")
    return count

def main():
    if len(sys.argv) < 2:
        base_dir = "/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/2024_25/arxiv_data/full_papers"
        print(f"Using default directory: {base_dir}")
    else:
        base_dir = sys.argv[1]
    
    # Check if user wants quick count or detailed analysis
    mode = sys.argv[2] if len(sys.argv) > 2 else "detailed"
    
    print("LaTeX Paper Counter")
    print("=" * 50)
    
    if mode.lower() in ['quick', 'q']:
        quick_count(base_dir)
    else:
        stats = count_papers_by_year(base_dir)
        
        # Optionally save statistics
        if stats and '--save' in sys.argv:
            import json
            with open('paper_statistics.json', 'w') as f:
                # Convert any non-serializable objects
                stats_clean = {
                    'total_files': stats['total_files'],
                    'year_counts': stats['year_counts'],
                    'month_counts': stats['month_counts'],
                    'file_extensions': stats['file_extensions'],
                    'total_dirs': stats['total_dirs'],
                    'avg_file_size': sum(stats['file_sizes'])/len(stats['file_sizes']) if stats['file_sizes'] else 0,
                    'median_file_size': sorted(stats['file_sizes'])[len(stats['file_sizes'])//2] if stats['file_sizes'] else 0
                }
                json.dump(stats_clean, f, indent=2)
            print(f"\nStatistics saved to paper_statistics.json")

if __name__ == "__main__":
    main()

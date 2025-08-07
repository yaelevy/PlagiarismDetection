#!/bin/bash

# Source the find_document function
source find_document_by_number.sh

extract_passages() {
    local source_id="$1"
    local suspicious_id="$2"
    local source_offset="$3"
    local source_length="$4"
    local suspicious_offset="$5"
    local suspicious_length="$6"
    local corpus_path="$7"
    
    echo "Looking for source: $source_id"
    echo "Looking for suspicious: $suspicious_id"
    
    # Find the source document
    source_file=$(find_document "$source_id" "source-document" "$corpus_path")
    if [ $? -ne 0 ]; then
        echo "Error: Source file not found: $source_id" >&2
        return 1
    fi
    echo "Found source: $source_file"
    
    # Find the suspicious document (add .txt if not present)
    if [[ "$suspicious_id" != *.txt ]]; then
        suspicious_file=$(find_document "${suspicious_id}.txt" "suspicious-document" "$corpus_path")
    else
        suspicious_file=$(find_document "$suspicious_id" "suspicious-document" "$corpus_path")
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: Suspicious file not found: $suspicious_id" >&2
        return 1
    fi
    echo "Found suspicious: $suspicious_file"
    
    echo -e "\n=== SOURCE PASSAGE ==="
    dd if="$source_file" bs=1 skip="$source_offset" count="$source_length" 2>/dev/null
    
    echo -e "\n\n=== SUSPICIOUS PASSAGE ==="
    dd if="$suspicious_file" bs=1 skip="$suspicious_offset" count="$suspicious_length" 2>/dev/null
    
    echo -e "\n\n=========================="
}

# Test with our known example
test_extraction() {
    local corpus_path="/Users/yaronot/Desktop/Data Courses/Plagiarism Detection/pan-plagiarism-corpus-2011-1/external-detection-corpus"
    
    echo "Testing extraction with known example..."
    extract_passages "source-document08792.txt" "suspicious-document00027" 10104 498 1714 484 "$corpus_path"
}

export -f extract_passages
export -f test_extraction

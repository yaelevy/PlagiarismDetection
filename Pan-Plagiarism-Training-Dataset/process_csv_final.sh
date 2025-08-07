#!/bin/bash

# Source the extraction functions
source extract_passages_final.sh

# Function to process a single CSV row
process_csv_row() {
    local csv_line="$1"
    local corpus_path="$2"
    
    # Parse CSV line (format: source_id,suspicious_id,type,obfuscation,source_offset,source_length,suspicious_offset,suspicious_length)
    csv_line_clean=$(echo "$csv_line" | tr -d '\r')
    IFS=',' read -r source_id suspicious_id type obfuscation source_offset source_length suspicious_offset suspicious_length <<< "$csv_line_clean"    
    echo "========================================"
    echo "Processing: $suspicious_id -> $source_id"
    echo "Type: $type, Obfuscation: $obfuscation"
    echo "Offsets: source[$source_offset:$source_length] suspicious[$suspicious_offset:$suspicious_length]"
    echo "========================================"
    
    extract_passages "$source_id" "$suspicious_id" "$source_offset" "$source_length" "$suspicious_offset" "$suspicious_length" "$corpus_path"
    
    echo -e "\n\n"
}

# Function to process specific row number from CSV
extract_row() {
    local row_number="$1"  # 1-based (1 = first data row, not header)
    local csv_file="$2"
    local corpus_path="$3"
    
    # Get the specific row (skip header + get row number)
    csv_line=$(tail -n +2 "$csv_file" | sed -n "${row_number}p")
    
    if [ -z "$csv_line" ]; then
        echo "Error: Row $row_number not found in CSV"
        return 1
    fi
    
    echo "=== EXTRACTING ROW $row_number FROM CSV ==="
    process_csv_row "$csv_line" "$corpus_path"
}

# Function to process first N rows
process_first_n_rows() {
    local n="$1"
    local csv_file="$2"
    local corpus_path="$3"
    
    echo "Processing first $n rows from $csv_file"
    tail -n +2 "$csv_file" | head -"$n" | while IFS= read -r line; do
        process_csv_row "$line" "$corpus_path"
        echo "=========================================="
    done
}

export -f process_csv_row
export -f extract_row
export -f process_first_n_rows

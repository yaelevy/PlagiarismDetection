#!/bin/bash

# Function to extract document number and calculate part directory
get_part_number() {
    local filename="$1"
    
    # Extract the numeric part from filenames like "source-document08792.txt" or "suspicious-document00027.txt"
    local number=$(echo "$filename" | sed 's/.*-document0*\([0-9]*\)\.txt$/\1/')
    
    if [ -z "$number" ] || [ "$number" = "$filename" ]; then
        echo "Error: Could not extract number from $filename" >&2
        return 1
    fi
    
    # Calculate part number: (number - 1) / 500 + 1
    # Part1: 1-500, Part2: 501-1000, etc.
    local part_num=$(( (number - 1) / 500 + 1 ))
    
    echo "$part_num"
    return 0
}

# Function to find a document using calculated part number
find_document() {
    local filename="$1"
    local doc_type="$2"  # "source-document" or "suspicious-document"
    local corpus_path="$3"
    
    # Get the part number for this file
    local part_num=$(get_part_number "$filename")
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    # Construct the expected path
    local file_path="$corpus_path/$doc_type/part${part_num}/$filename"
    
    if [ -f "$file_path" ]; then
        echo "$file_path"
        return 0
    else
        echo "File not found: $file_path" >&2
        return 1
    fi
}

export -f get_part_number
export -f find_document

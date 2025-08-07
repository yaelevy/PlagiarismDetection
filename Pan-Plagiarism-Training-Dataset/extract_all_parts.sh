#!/bin/bash

# Add CSV header
echo "source_id,suspicious_id,type,obfuscation,source_offset,source_length,suspicious_offset,suspicious_length"

# Process all parts
for part_dir in part*; do
if [ -d "$part_dir" ]; then
echo "Processing $part_dir..." >&2 # Error stream so it doesn't go to CSV

# Change to part directory
cd "$part_dir"

# Process all XML files that contain source_reference
for file in $(grep -l "source_reference" *.xml 2>/dev/null); do
suspicious_id=$(basename "$file" .xml)

# Extract plagiarism features using grep and sed
grep 'feature name="plagiarism"' "$file" | while read line; do
# Extract attributes using sed
type=$(echo "$line" | sed -n 's/.*type="\([^"]*\)".*/\1/p')
obfuscation=$(echo "$line" | sed -n 's/.*obfuscation="\([^"]*\)".*/\1/p')
this_offset=$(echo "$line" | sed -n 's/.*this_offset="\([^"]*\)".*/\1/p')
this_length=$(echo "$line" | sed -n 's/.*this_length="\([^"]*\)".*/\1/p')
source_ref=$(echo "$line" | sed -n 's/.*source_reference="\([^"]*\)".*/\1/p')
source_offset=$(echo "$line" | sed -n 's/.*source_offset="\([^"]*\)".*/\1/p')
source_length=$(echo "$line" | sed -n 's/.*source_length="\([^"]*\)".*/\1/p')

# Only output if we have source_reference (complete mapping)
if [ ! -z "$source_ref" ]; then
echo "$source_ref,$suspicious_id,$type,$obfuscation,$source_offset,$source_length,$this_offset,$this_length"
fi
done
done

# Go back to parent directory
cd ..
fi
done

echo "Extraction completed!" >&2
EOF

chmod +x extract_all_parts.sh

#!/bin/bash

# Directory to check (can be customized)
DIR="."

# Find all C++ source/header files
FILES=$(find "$DIR" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" -o -name "*.h" \))

NOT_FORMATTED=()

for file in $FILES; do
    # Check formatting using clang-format
    if ! diff -q "$file" <(clang-format "$file") >/dev/null; then
        NOT_FORMATTED+=("$file")
    fi
done

if [ ${#NOT_FORMATTED[@]} -eq 0 ]; then
    echo "🌟 All files are properly formatted. Good job! 🌟"
else
    echo "The following files are not properly formatted: 😟"
    for f in "${NOT_FORMATTED[@]}"; do
        echo "$f"
    done
    exit 1
fi
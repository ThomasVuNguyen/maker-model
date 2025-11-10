#!/bin/bash
# Simple OpenSCAD Syntax Validator
# Basic syntax checking without requiring X server

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TOTAL=0
PASSED=0
FAILED=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OpenSCAD Syntax Checker (Basic)${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Basic syntax checks
check_syntax() {
    local file="$1"
    local filename=$(basename "$file")
    local errors=()

    TOTAL=$((TOTAL + 1))
    echo -n "Checking ${filename}... "

    # Read file content
    content=$(<"$file")

    # Skip generated comments
    code=$(echo "$content" | grep -v "^//" | grep -v "^$")

    # Check for common syntax errors
    local has_error=0

    # Check for balanced braces
    open_braces=$(echo "$code" | grep -o "{" | wc -l)
    close_braces=$(echo "$code" | grep -o "}" | wc -l)
    if [ "$open_braces" != "$close_braces" ]; then
        errors+=("Unbalanced braces: $open_braces open, $close_braces close")
        has_error=1
    fi

    # Check for balanced parentheses
    open_parens=$(echo "$code" | grep -o "(" | wc -l)
    close_parens=$(echo "$code" | grep -o ")" | wc -l)
    if [ "$open_parens" != "$close_parens" ]; then
        errors+=("Unbalanced parentheses: $open_parens open, $close_parens close")
        has_error=1
    fi

    # Check for balanced brackets
    open_brackets=$(echo "$code" | grep -o "\[" | wc -l)
    close_brackets=$(echo "$code" | grep -o "\]" | wc -l)
    if [ "$open_brackets" != "$close_brackets" ]; then
        errors+=("Unbalanced brackets: $open_brackets open, $close_brackets close")
        has_error=1
    fi

    # Check for empty file (only comments)
    if [ -z "$code" ]; then
        errors+=("File contains only comments or is empty")
        has_error=1
    fi

    # Check for basic OpenSCAD keywords (at least one should exist)
    if ! echo "$code" | grep -qE '(module|function|sphere|cube|cylinder|union|difference|intersection|translate|rotate|scale)'; then
        errors+=("No OpenSCAD keywords found")
        has_error=1
    fi

    # Check for non-ASCII garbage
    if echo "$code" | grep -qP '[^\x00-\x7F]'; then
        errors+=("Contains non-ASCII characters")
        has_error=1
    fi

    # Report results
    if [ $has_error -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASSED=$((PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        for error in "${errors[@]}"; do
            echo -e "  ${RED}└─${NC} $error"
        done
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Test directory
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory|file.scad>"
    exit 1
fi

target="$1"

if [ -d "$target" ]; then
    echo -e "Testing files in: ${BLUE}$target${NC}\n"
    shopt -s nullglob
    files=("$target"/*.scad)

    if [ ${#files[@]} -eq 0 ]; then
        echo -e "${YELLOW}No .scad files found${NC}"
        exit 0
    fi

    for file in "${files[@]}"; do
        check_syntax "$file"
    done
elif [ -f "$target" ]; then
    check_syntax "$target"
else
    echo -e "${RED}Error: '$target' not found${NC}"
    exit 1
fi

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Total:  $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $TOTAL -gt 0 ]; then
    pass_rate=$(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")
    echo -e "\nPass Rate: ${pass_rate}%"
fi

echo -e "${BLUE}========================================${NC}\n"

if [ $FAILED -gt 0 ]; then
    exit 1
fi

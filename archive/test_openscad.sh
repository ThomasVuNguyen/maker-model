#!/bin/bash
# OpenSCAD Code Validator
# Tests whether OpenSCAD files compile without errors

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OPENSCAD_BIN="openscad"
TIMEOUT=10  # seconds

# Statistics
TOTAL=0
PASSED=0
FAILED=0
WARNINGS=0

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}OpenSCAD Code Validator${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to test a single .scad file
test_scad_file() {
    local file="$1"
    local filename=$(basename "$file")

    TOTAL=$((TOTAL + 1))

    echo -n "Testing ${filename}... "

    # Try to compile and render the file to 3MF format
    # This validates both syntax and geometry without needing X server
    temp_output="/tmp/openscad_test_$$.3mf"
    output=$(timeout ${TIMEOUT}s ${OPENSCAD_BIN} --render -o "$temp_output" "$file" 2>&1)
    exit_code=$?

    # Clean up temp file
    rm -f "$temp_output"

    if [ $exit_code -eq 0 ]; then
        # Check for warnings in output
        if echo "$output" | grep -qi "warning"; then
            echo -e "${YELLOW}⚠ PASS (with warnings)${NC}"
            WARNINGS=$((WARNINGS + 1))
            echo "$output" | grep -i "warning" | sed 's/^/  /'
        else
            echo -e "${GREEN}✓ PASS${NC}"
        fi
        PASSED=$((PASSED + 1))
        return 0
    elif [ $exit_code -eq 124 ]; then
        echo -e "${RED}✗ FAIL (timeout)${NC}"
        FAILED=$((FAILED + 1))
        return 1
    else
        echo -e "${RED}✗ FAIL${NC}"
        FAILED=$((FAILED + 1))

        # Show compilation errors
        if [ ! -z "$output" ]; then
            echo "$output" | grep -E "(ERROR|error|PARSE)" | head -5 | sed 's/^/  /'
        fi
        return 1
    fi
}

# Function to test all .scad files in a directory
test_directory() {
    local dir="$1"

    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: Directory '$dir' not found${NC}"
        exit 1
    fi

    echo -e "Testing files in: ${BLUE}$dir${NC}\n"

    # Find all .scad files
    shopt -s nullglob
    files=("$dir"/*.scad)

    if [ ${#files[@]} -eq 0 ]; then
        echo -e "${YELLOW}No .scad files found in directory${NC}"
        exit 0
    fi

    # Test each file
    for file in "${files[@]}"; do
        test_scad_file "$file"
    done
}

# Function to test a single file
test_single_file() {
    local file="$1"

    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: File '$file' not found${NC}"
        exit 1
    fi

    if [[ ! "$file" =~ \.scad$ ]]; then
        echo -e "${RED}Error: File must have .scad extension${NC}"
        exit 1
    fi

    test_scad_file "$file"
}

# Function to show summary
show_summary() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Test Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "Total:    $TOTAL"
    echo -e "${GREEN}Passed:   $PASSED${NC}"
    echo -e "${RED}Failed:   $FAILED${NC}"
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"

    # Calculate pass rate
    if [ $TOTAL -gt 0 ]; then
        pass_rate=$(awk "BEGIN {printf \"%.1f\", ($PASSED/$TOTAL)*100}")
        echo -e "\nPass Rate: ${pass_rate}%"
    fi

    echo -e "${BLUE}========================================${NC}\n"

    # Exit with failure if any tests failed
    if [ $FAILED -gt 0 ]; then
        exit 1
    fi
}

# Check if OpenSCAD is installed
if ! command -v ${OPENSCAD_BIN} &> /dev/null; then
    echo -e "${RED}Error: OpenSCAD not found${NC}"
    echo "Please install OpenSCAD: sudo apt install openscad"
    exit 1
fi

# Main script logic
if [ $# -eq 0 ]; then
    echo "Usage: $0 <file.scad|directory>"
    echo ""
    echo "Examples:"
    echo "  $0 test.scad                    # Test single file"
    echo "  $0 eval_outputs/eval_*/         # Test directory"
    echo "  $0 eval_outputs/eval_20251110_*/  # Test specific eval folder"
    exit 1
fi

# Parse arguments
for arg in "$@"; do
    if [ -d "$arg" ]; then
        test_directory "$arg"
    elif [ -f "$arg" ]; then
        test_single_file "$arg"
    else
        echo -e "${YELLOW}Warning: Skipping '$arg' (not found)${NC}"
    fi
done

# Show summary if we tested anything
if [ $TOTAL -gt 0 ]; then
    show_summary
fi

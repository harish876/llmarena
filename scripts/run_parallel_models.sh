#!/bin/bash

# GNU parallel script to run matharena for multiple models
# Usage: ./scripts/run_parallel_models.sh --comp <competition> [--models-file <file>] [-j <jobs>] [--preview]
# Example: ./scripts/run_parallel_models.sh --comp aime/aime_2025
# Example: ./scripts/run_parallel_models.sh --comp agieval/lsat_ar --preview
# Example: ./scripts/run_parallel_models.sh --comp aime/aime_2025 --models-file configs/models/include.txt -j 4

set -e

# Default values
MODELS_FILE=""
JOBS=""
COMP=""
PREVIEW=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --comp)
            COMP="$2"
            shift 2
            ;;
        --models-file)
            MODELS_FILE="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --preview)
            PREVIEW=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --comp <competition> [--models-file <file>] [-j <jobs>] [--preview]"
            echo ""
            echo "Arguments:"
            echo "  --comp <competition>     Competition name (e.g., aime/aime_2025, agieval/lsat_ar)"
            echo "  --models-file <file>     File containing model names (default: configs/competitions/<category>/include.txt)"
            echo "  -j, --jobs <jobs>        Number of parallel jobs (default: all available cores)"
            echo "  --preview                Run with only first 5 problems (uses --max-problems 5)"
            echo ""
            echo "Example:"
            echo "  $0 --comp aime/aime_2025"
            echo "  $0 --comp agieval/lsat_ar --preview"
            echo "  $0 --comp aime/aime_2025 --models-file configs/models/include.txt -j 4"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if competition is provided
if [[ -z "$COMP" ]]; then
    echo "Error: --comp argument is required"
    echo "Use --help for usage information"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Extract category from competition (e.g., "agieval" from "agieval/lsat_ar")
COMP_CATEGORY="${COMP%%/*}"

# Set default models file if not provided
if [[ -z "$MODELS_FILE" ]]; then
    # Try competition-specific include.txt first
    COMP_MODELS_FILE="configs/competitions/${COMP_CATEGORY}/include.txt"
    if [[ -f "$COMP_MODELS_FILE" ]]; then
        MODELS_FILE="$COMP_MODELS_FILE"
    else
        # Fall back to global models include.txt
        MODELS_FILE="configs/models/include.txt"
    fi
fi

# Check if models file exists
if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: Models file not found: $MODELS_FILE"
    exit 1
fi

# Check if GNU parallel is installed
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU parallel is not installed"
    echo "Install it with: sudo apt-get install parallel"
    exit 1
fi

# Build parallel command
# Default to number of CPU cores if not specified, but allow override
PARALLEL_CMD="parallel"
CPU_CORES=$(nproc)

if [[ -n "$JOBS" ]]; then
    PARALLEL_CMD="$PARALLEL_CMD -j $JOBS"
    JOBS_DISPLAY="$JOBS"
else
    # Default to number of CPU cores to avoid overwhelming the system
    PARALLEL_CMD="$PARALLEL_CMD -j $CPU_CORES"
    JOBS_DISPLAY="$CPU_CORES (auto)"
fi

# Add progress bar and keep output from different jobs separate
PARALLEL_CMD="$PARALLEL_CMD --progress --tag"

# Create log file with timestamp and competition name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMP_NAME=$(echo "$COMP" | tr '/' '_')
LOG_DIR="logs/parallel_runs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${COMP_NAME}_${TIMESTAMP}.log"

# Filter out commented lines and empty lines
# Supports both # and // comment styles, and inline comments
FILTERED_MODELS=$(mktemp)
trap "rm -f $FILTERED_MODELS" EXIT

# Filter processing:
# 1. Skip lines that start with # or // (full-line comments)
# 2. Remove inline // comments (everything from // to end of line)
# 3. Remove inline # comments (everything from # to end of line)
# 4. Trim leading and trailing whitespace
# 5. Skip empty lines
awk '{
    original = $0
    # Skip lines that start with comment markers
    if (match(original, /^[[:space:]]*[#\/]/)) {
        next
    }
    # Remove inline comments
    gsub(/[[:space:]]*\/\/.*$/, "")
    gsub(/[[:space:]]*#.*$/, "")
    # Trim whitespace
    gsub(/^[[:space:]]+/, "")
    gsub(/[[:space:]]+$/, "")
    # Print if not empty
    if (length($0) > 0) {
        print
    }
}' "$MODELS_FILE" > "$FILTERED_MODELS"

ACTIVE_MODELS=$(wc -l < "$FILTERED_MODELS" | tr -d ' ')

if [[ $ACTIVE_MODELS -eq 0 ]]; then
    echo "Error: No active models found in $MODELS_FILE (all lines are commented or empty)"
    exit 1
fi

echo "Running parallel execution with:"
echo "  Competition: $COMP"
echo "  Models file: $MODELS_FILE"
echo "  Parallel jobs: $JOBS_DISPLAY"
echo "  Active models: $ACTIVE_MODELS"
if [[ "$PREVIEW" == true ]]; then
    echo "  Preview mode: Yes (max 5 problems)"
fi
echo "  Note: Each model runs in a separate process"
echo "  Log file: $LOG_FILE"
echo ""

# Write header to log file
{
    echo "=========================================="
    echo "Parallel Run Started: $(date)"
    echo "Competition: $COMP"
    echo "Models file: $MODELS_FILE"
    echo "Parallel jobs: $JOBS_DISPLAY"
    echo "Active models: $ACTIVE_MODELS"
    if [[ "$PREVIEW" == true ]]; then
        echo "Preview mode: Yes (max 5 problems)"
    fi
    echo "=========================================="
    echo ""
} | tee "$LOG_FILE"

# Run parallel execution and write all output to log file
# Read model names from filtered file and run the command for each
# Conditionally add --max-problems 5 if preview mode is enabled
if [[ "$PREVIEW" == true ]]; then
    $PARALLEL_CMD --trim rl \
        uv run python scripts/run.py --comp "$COMP" --models {} --max-problems 5 \
        :::: "$FILTERED_MODELS" 2>&1 | tee -a "$LOG_FILE"
else
    $PARALLEL_CMD --trim rl \
        uv run python scripts/run.py --comp "$COMP" --models {} \
        :::: "$FILTERED_MODELS" 2>&1 | tee -a "$LOG_FILE"
fi

# Write footer to log file
{
    echo ""
    echo "=========================================="
    echo "Parallel Run Completed: $(date)"
    echo "=========================================="
} | tee -a "$LOG_FILE"

echo ""
echo "All jobs completed! Logs saved to: $LOG_FILE"


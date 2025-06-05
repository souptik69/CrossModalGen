#!/bin/bash

#SBATCH --job-name=text_query_test
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=1:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/Slurm/text_query_test/test_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/Slurm/text_query_test/test_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New

# Create output directories for logs
mkdir -p $SRCDIR/Experiments/Misc/Slurm/text_query_test

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Define parameters
PICKLE_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl"
CSV_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsoundCategories2Prompts.csv"
META_CSV_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data/vggsound-avel100k-new.csv"
OUTPUT_DIR="$SRCDIR/Experiments/Misc/text_query_test_results"
NUM_SAMPLES=150

# Create output directory
mkdir -p $OUTPUT_DIR

# Print job information
echo "============================================"
echo "TEXT QUERY PIPELINE TEST"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "============================================"
echo ""

# Print parameters
echo "Parameters:"
echo "  Pickle path: $PICKLE_PATH"
echo "  CSV path: $CSV_PATH"  
echo "  Meta CSV path: $META_CSV_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of samples: $NUM_SAMPLES"
echo ""

# Check if input files exist
echo "Checking input files..."
if [ -f "$PICKLE_PATH" ]; then
    echo "  ✓ Pickle file found: $PICKLE_PATH"
    echo "    Size: $(ls -lh "$PICKLE_PATH" | awk '{print $5}')"
else
    echo "  ✗ Pickle file NOT found: $PICKLE_PATH"
    exit 1
fi

if [ -f "$CSV_PATH" ]; then
    echo "  ✓ CSV file found: $CSV_PATH"
    echo "    Lines: $(wc -l < "$CSV_PATH")"
else
    echo "  ✗ CSV file NOT found: $CSV_PATH"
    exit 1
fi

if [ -f "$META_CSV_PATH" ]; then
    echo "  ✓ Meta CSV file found: $META_CSV_PATH"
    echo "    Lines: $(wc -l < "$META_CSV_PATH")"
else
    echo "  ✗ Meta CSV file NOT found: $META_CSV_PATH"
    exit 1
fi

echo ""

# Run the test script
echo "Starting text query pipeline test..."
echo "============================================"

python $SRCDIR/Experiments/Misc/test_text_query_pipeline.py \
    --pickle_path "$PICKLE_PATH" \
    --csv_path "$CSV_PATH" \
    --meta_csv_path "$META_CSV_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples $NUM_SAMPLES

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Test completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================"

# Display results summary if successful
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "RESULTS SUMMARY:"
    echo "================"
    
    # Check if output file exists and display key findings
    RESULTS_FILE="$OUTPUT_DIR/detailed_text_query_analysis.txt"
    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo "Key findings from analysis:"
        echo ""
        
        # Extract key metrics
        echo "Data loading:"
        grep -E "(id2idx entries|Label2prompt entries|Meta dataset size)" "$RESULTS_FILE" | sed 's/^/  /'
        
        echo ""
        echo "Processing results:"
        grep -E "(Total samples|Average retention rate|Empty samples|Processing time)" "$RESULTS_FILE" | sed 's/^/  /'
        
        echo ""
        echo "Query tensor info:"
        grep -E "(Shape|Non-zero percentage|Query lengths)" "$RESULTS_FILE" | sed 's/^/  /'
        
        echo ""
        echo "Sample retention rates:"
        grep -A1 "Sample [0-9]:" "$RESULTS_FILE" | grep "Retention rate" | head -10 | sed 's/^/  /'
        
        echo ""
        echo "Full detailed results saved to:"
        echo "  $RESULTS_FILE"
        
        # Show file size
        echo "  File size: $(ls -lh "$RESULTS_FILE" | awk '{print $5}')"
    else
        echo "⚠️  Results file not found. Check logs for errors."
    fi
    
    # List all output files
    echo ""
    echo "All output files:"
    ls -la "$OUTPUT_DIR" | sed 's/^/  /'
    
else
    echo ""
    echo "❌ Test failed! Check the error log for details."
    echo "Error log: /project/ag-jafra/Souptik/CMG_New/Experiments/Misc/Slurm/text_query_test/test_${SLURM_JOB_ID}.err"
fi

echo ""
echo "Log files:"
echo "  Output: /project/ag-jafra/Souptik/CMG_New/Experiments/Misc/Slurm/text_query_test/test_${SLURM_JOB_ID}.out"
echo "  Error:  /project/ag-jafra/Souptik/CMG_New/Experiments/Misc/Slurm/text_query_test/test_${SLURM_JOB_ID}.err"
#!/bin/bash

#SBATCH --job-name=transformers_specific
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=0:30:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/specific_test/transformers_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/specific_test/transformers_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New

# Create output directories for logs
mkdir -p $SRCDIR/Experiments/Slurm/specific_test

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Define parameters
PICKLE_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/CMG/cnt.pkl"
OUTPUT_DIR="$SRCDIR/Experiments/Misc/transformers_specific_embeddings"
mkdir -p $OUTPUT_DIR

# Run the script
echo "Starting Transformers BERT embedding extraction for specific sentences..."
python $SRCDIR/Experiments/Misc/test_transformers_modified.py \
    --pickle_path $PICKLE_PATH \
    --output_dir $OUTPUT_DIR

echo "Embedding extraction complete! Results are in $OUTPUT_DIR"

# Display a quick summary
if [ -f "$OUTPUT_DIR/summary.txt" ]; then
    echo -e "\nSummary of embeddings:"
    grep "First token embedding" "$OUTPUT_DIR/summary.txt" | head -n 3
    echo -e "\nFilter statistics:"
    grep "Filtered tokens:" "$OUTPUT_DIR/summary.txt" | head -n 6
    echo -e "\nProcessing time:"
    grep "Total processing time" "$OUTPUT_DIR/summary.txt"
else
    echo "Summary file not found. Check logs for errors."
fi
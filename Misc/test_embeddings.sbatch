#!/bin/bash

#SBATCH --job-name=transformers_test
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=0:30:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/transformers_test/test_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/transformers_test/test_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New

# Create output directories for logs
mkdir -p $SRCDIR/Experiments/Slurm/transformers_test

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Define parameters
PROMPTS_CSV="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/vggsound40k/data/vggsoundCategories2Prompts.csv"
PICKLE_PATH="/project/ag-jafra/Souptik/CMG_New/CMG/cnt.pkl"
OUTPUT_DIR="$SRCDIR/Experiments/Misc/transformers_test_results"
NUM_SAMPLES=3  # Number of sentences to test

# Make sure transformers is installed
pip install transformers --no-cache-dir

# Run the test script
echo "Starting Transformers BERT embedding test with $NUM_SAMPLES sample sentences..."
python $SRCDIR/Experiments/Misc/test_embeddings.py \
    --prompts_csv $PROMPTS_CSV \
    --pickle_path $PICKLE_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES

echo "Transformers embedding test complete! Results are in $OUTPUT_DIR"
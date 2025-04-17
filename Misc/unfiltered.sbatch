#!/bin/bash
#SBATCH --job-name=unfiltered_emb
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=0:30:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/unfiltered/unfiltered_emb_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/unfiltered/unfiltered_emb_%j.err


# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New

mkdir -p $SRCDIR/Experiments/Slurm/unfiltered


# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Set paths
OUTPUT_DIR="$SRCDIR/Experiments/Misc/embeddings_unfiltered"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the script
python $SRCDIR/Experiments/Misc/unfiltered_embeddings.py \
    --output_dir $OUTPUT_DIR

echo "Unfiltered embeddings completed."
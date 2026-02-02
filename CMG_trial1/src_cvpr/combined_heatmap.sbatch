#!/bin/bash

#SBATCH --job-name=compare_codaar_micu
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=5:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization/comparison_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization/comparison_%j.err

# Create output directory
mkdir -p /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization

echo "======================================================================="
echo "CODEBOOK USAGE COMPARISON: CoDAAR vs MICU"
echo "======================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo ""
echo "This script will:"
echo "  1. Load CoDAAR model (800-vector SPLIT codebook)"
echo "  2. Load MICU model (400-vector SHARED codebook)"
echo "  3. Process VGGSound dataset through both models"
echo "  4. Generate 4-row comparison heatmap"
echo "======================================================================="
echo ""

# Activate conda environment (using CoDAAR's environment)
source ~/.bashrc
# conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_final
conda activate /project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/CMG_final
# Verify environment
echo "Environment Info:"
echo "  Python: $(which python)"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run comparison analysis
echo "Starting comparison analysis..."
python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/balanced_heatmap.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "COMPARISON COMPLETE - SUCCESS!"
    echo "======================================================================="
    echo "End Time: $(date)"
    echo ""
    echo "Output Files Generated:"
    echo "  - Comparison_CoDAAR_vs_MICU_heatmap.png"
    echo "  - Comparison_statistics.txt"
    echo ""
    echo "Location:"
    echo "  /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Analysis/Codebook_Utilization/"
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "ERROR: Comparison failed with exit code $?"
    echo "======================================================================="
    echo "End Time: $(date)"
    echo "Check error log for details"
    exit 1
fi
#!/bin/bash

#SBATCH --job-name=Comprehensive_Padding_Analysis
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=04:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/01_padding_analysis/padding_analysis_output_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/01_padding_analysis/padding_analysis_error_%j.err

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "========================================="

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories
mkdir -p $SRCDIR/CMG_trial1/slurm/01_padding_analysis
mkdir -p $SRCDIR/CMG_trial1/Results/padding_analysis_seqlen

# Source bashrc and activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

echo "Starting comprehensive padding analysis..."
echo "Analysis will cover all MOSEI and MOSI splits with sentiment bin analysis"
echo "Expected runtime: 1-3 hours depending on data size"
echo ""

# Change to the source directory for proper imports
cd $SRCDIR/CMG_trial1/src

# Run the comprehensive padding analysis
python -u comprehensive_padding_analysis.py

echo ""
echo "========================================="
echo "Padding analysis completed!"
echo "Results saved to: $SRCDIR/CMG_trial1/Results/padding_analysis_seqlen/"
echo "Check the following files for results:"
echo "  - padding_analysis_summary.csv (main statistics)"
echo "  - padding_analysis_detailed_report.txt (detailed report)"  
echo "  - complete_padding_analysis.json (full raw data)"
echo "End Time: $(date)"
echo "========================================="
#!/bin/bash

#SBATCH --job-name=Codebook_Sentiment_Analysis
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --time=15:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/0_Check_Codebook_Window/codebook_sentiment_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm1/0_Check_Codebook_Window/codebook_sentiment_%j.err



# Create necessary directories for organized output management
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments
mkdir -p $SRCDIR/CMG_trial1/slurm1/0_Check_Codebook_Window
mkdir -p $SRCDIR/Misc/codebook_sentiment_analysis_10

# Activate the conda environment with all required dependencies
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Print comprehensive job information for debugging and tracking purposes
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python Version: $(python --version)"
echo "Conda Environment: $CONDA_DEFAULT_ENV"
echo "GPU Information: $(nvidia-smi -L)"
echo "==========================================="

# =============================================================================
# MODEL CHECKPOINT CONFIGURATION
# 
# This section allows you to easily switch between different trained models
# for comparative analysis. Each model may have learned different quantization
# patterns, so analyzing multiple checkpoints can reveal interesting insights.
# =============================================================================

# OPTION 1: Supervised Model Analysis (DEFAULT)
# This model was trained with sentiment labels and should show clear
# quantization patterns that correlate with emotional content
CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models_3/ContentAware_MOSEI_unsupervised_TAV_contrastive_64dim_window/checkpoint/MOSEI-model-unsupervised-23.pt"
ANALYSIS_TYPE="ContentAware_MOSEI_supervised_TAV_contrastive_64dim_window_23"

# OPTION 2: Unsupervised Model Analysis (UNCOMMENT TO USE)
# This model learned representations without explicit sentiment supervision
# Comparing its quantization patterns with the supervised model reveals
# whether sentiment structure emerges naturally from multimodal data
# CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models/unsupervised/checkpoint/MOSEI-model-unsupervised-4.pt"
# ANALYSIS_TYPE="unsupervised"

# OPTION 3: LSTM-Enhanced Model Analysis (UNCOMMENT TO USE)
# This variant incorporates LSTM temporal modeling which may create
# different quantization patterns due to enhanced sequence understanding
# CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/MOSEI_Models2/LSTM_seq50_100_supervised_reg_TAV/checkpoint/MOSEI-model-16.pt"
# ANALYSIS_TYPE="supervised_lstm"

# =============================================================================
# ANALYSIS CONFIGURATION PARAMETERS
# 
# These parameters control the depth and scope of the analysis. Adjusting
# them allows you to balance computational cost against analytical detail.
# =============================================================================

# Output directory with timestamp to prevent conflicts between runs
# This organization allows you to compare results from different experiments
OUTPUT_DIR="/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/codebook_sentiment_analysis_10/${ANALYSIS_TYPE}"

# Sampling parameters that determine analysis comprehensiveness
SAMPLES_PER_LABEL=5        # More samples per bin provide better statistical validity
NUM_SENTIMENT_BINS=7       # Seven bins capture the full emotional spectrum well
BATCH_SIZE=16             # Smaller batches during analysis reduce memory pressure
MAX_SEQ_LEN=50           # Matches training configuration for consistency
NUM_WORKERS=8            # Fewer workers during analysis prevent resource conflicts

# Model architecture parameters that must match training configuration exactly
# These values define the expected input and output dimensions
AUDIO_DIM=74             # Standard audio feature dimension for MOSEI/MOSI
VIDEO_DIM=35             # Standard video feature dimension for MOSEI/MOSI  
TEXT_DIM=300             # Standard text embedding dimension for MOSEI/MOSI
N_EMBEDDINGS=256         # Codebook size that determines quantization granularity
EMBEDDING_DIM=64        # Dimension of individual semantic representations



echo "Starting comprehensive codebook-sentiment analysis..."
echo "Configuration Summary:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Analysis type: $ANALYSIS_TYPE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Samples per sentiment bin: $SAMPLES_PER_LABEL"
echo "  Number of sentiment bins: $NUM_SENTIMENT_BINS"
echo "  Model architecture: Audio=${AUDIO_DIM}D, Video=${VIDEO_DIM}D, Text=${TEXT_DIM}D"
echo "  Codebook configuration: ${N_EMBEDDINGS} vectors of ${EMBEDDING_DIM} dimensions"

# =============================================================================
# PRE-EXECUTION VALIDATION
# 
# These checks prevent wasted computational time by verifying prerequisites
# before starting the intensive analysis process.
# =============================================================================

# Verify that the specified checkpoint file actually exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "==========================================="
    echo "ERROR: Checkpoint file not found!"
    echo "Expected location: $CHECKPOINT_PATH"
    echo ""
    echo "Please verify:"
    echo "1. The checkpoint path is correct"
    echo "2. The file has been successfully saved from training"
    echo "3. You have proper file system permissions"
    echo "==========================================="
    exit 1
fi

# Verify that the Python script exists in the expected location
PYTHON_SCRIPT_PATH="$SRCDIR/CMG_trial1/src/codebook_sentiment_analysis.py"
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "==========================================="
    echo "ERROR: Python analysis script not found!"
    echo "Expected location: $PYTHON_SCRIPT_PATH"
    echo ""
    echo "Please ensure the Python script has been saved to the correct directory."
    echo "==========================================="
    exit 1
fi

echo "All prerequisites verified successfully."
echo "Checkpoint file confirmed: $(ls -lh $CHECKPOINT_PATH)"
echo "Python script confirmed: $(ls -lh $PYTHON_SCRIPT_PATH)"

# Navigate to the source directory where the Python script is located
# This ensures proper module import paths for the analysis
cd $SRCDIR/CMG_trial1/src

# =============================================================================
# MAIN ANALYSIS EXECUTION
# 
# This command runs the comprehensive analysis with all specified parameters.
# The extensive parameter list provides full control over the analysis process.
# =============================================================================

echo ""
echo "Launching comprehensive codebook-sentiment analysis..."
echo "This process will:"
echo "1. Sample representative data from each sentiment bin"
echo "2. Extract semantic representations from temporal attention modules"  
echo "3. Perform detailed quantization analysis for each modality"
echo "4. Analyze padding vs content timestep quantization patterns"
echo "5. Generate comprehensive statistical summaries"
echo ""

python codebook_sentiment_analysis.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --samples_per_label $SAMPLES_PER_LABEL \
    --num_sentiment_bins $NUM_SENTIMENT_BINS \
    --batch_size $BATCH_SIZE \
    --max_seq_len $MAX_SEQ_LEN \
    --num_workers $NUM_WORKERS \
    --audio_dim $AUDIO_DIM \
    --video_dim $VIDEO_DIM \
    --text_dim $TEXT_DIM \
    --n_embeddings $N_EMBEDDINGS \
    --embedding_dim $EMBEDDING_DIM

# =============================================================================
# POST-EXECUTION ANALYSIS AND REPORTING
# 
# This section provides immediate feedback about job completion and guides
# you toward the next steps in your research workflow.
# =============================================================================

# Capture the exit code to determine if the analysis completed successfully
ANALYSIS_EXIT_CODE=$?

echo ""
echo "==========================================="

if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Codebook-sentiment analysis completed successfully!"
    echo ""
    echo "Analysis Results Available:"
    echo "  Primary output directory: $OUTPUT_DIR"
    echo "  Summary report: $OUTPUT_DIR/codebook_sentiment_analysis_summary.txt"
    echo "  Detailed logs: Check the SLURM output files"
    echo ""
    echo "Recommended Next Steps:"
    echo "1. Review the comprehensive summary report for high-level patterns"
    echo "2. Examine detailed logs for specific quantization behaviors"
    echo "3. Compare results across different sentiment bins"
    echo "4. Analyze codebook usage patterns for insights into model specialization"
    echo ""
    echo "Files created:"
    if [ -d "$OUTPUT_DIR" ]; then
        echo "$(find $OUTPUT_DIR -type f -name '*.txt' | wc -l) analysis files generated"
        echo "Total output size: $(du -sh $OUTPUT_DIR | cut -f1)"
    fi
    
else
    echo "ERROR: Analysis failed with exit code $ANALYSIS_EXIT_CODE"
    echo ""
    echo "Troubleshooting Steps:"
    echo "1. Check the SLURM error log for detailed error messages"
    echo "2. Verify that the model checkpoint is compatible with the specified architecture"
    echo "3. Ensure sufficient GPU memory is available"
    echo "4. Confirm that all required Python dependencies are installed"
    echo ""
    echo "Common Issues:"
    echo "- Model architecture mismatch between checkpoint and configuration"
    echo "- GPU memory exhaustion (try reducing batch_size)"
    echo "- Missing or incompatible Python packages"
    echo "- Network connectivity issues when loading datasets"
fi

echo "Job completion time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "==========================================="

# Exit with the same code as the Python script to propagate success/failure status
exit $ANALYSIS_EXIT_CODE
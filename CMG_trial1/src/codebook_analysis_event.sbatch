#!/bin/bash

#SBATCH --job-name=Codebook_Event_Analysis
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --time=12:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_Codebook_Events/codebook_events_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Check_Codebook_Events/codebook_events_%j.err

# Create necessary directories for organized output management
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments
mkdir -p $SRCDIR/CMG_trial1/slurm/0_Check_Codebook_Events
mkdir -p $SRCDIR/Misc/codebook_event_analysis

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
# DATASET AND MODEL TYPE CONFIGURATION
#
# Choose which dataset and model architecture to analyze. Each combination
# may reveal different quantization patterns based on the complexity of 
# audio-visual events and the model's capacity.
# =============================================================================

# DATASET SELECTION (CHOOSE ONE)
# DATASET="avvp"           # Options: "ave" or "avvp"
DATASET="ave"          # Uncomment for AVE dataset analysis

# MODEL TYPE SELECTION (CHOOSE ONE)
MODEL_TYPE="AVT"         # Options: "AV" or "AVT" 
# MODEL_TYPE="AV"        # Uncomment for Audio-Video only model

# =============================================================================
# MODEL CHECKPOINT CONFIGURATION
#
# Different pretrained models will show varying quantization patterns.
# AVT models (with text) may show different codebook usage compared to AV models.
# =============================================================================

if [ "$MODEL_TYPE" = "AVT" ]; then
    # OPTION 1: AVT Model - Best performing text-enhanced model (DEFAULT)
    CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVT_model/Best_Text_CPC_noNoise/40k/checkpoint/DCID-model-5.pt"
    ANALYSIS_TYPE="AVT_40k"
    
    # OPTION 2: AVT Model - 40k dataset variant (UNCOMMENT TO USE)
    # CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/AVT_model/Best_Text_CPC_noNoise/40k/checkpoint/DCID-model-5.pt"
    # ANALYSIS_TYPE="AVT_40k_TextCPC_noNoise"
    
else
    # AV Model checkpoints
    # OPTION 3: AV Model - Standard audio-video model (UNCOMMENT TO USE)
    CHECKPOINT_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Novel_Model_Final/Best_FixMetaModel_AV_final/40k/checkpoint/DCID-model-5.pt"
    ANALYSIS_TYPE="AV_40k"
fi

# =============================================================================
# DATASET PATH CONFIGURATION
#
# Configure paths based on the selected dataset. Each dataset has different
# storage formats and organizational structures.
# =============================================================================

if [ "$DATASET" = "ave" ]; then
    # AVE Dataset Configuration
    AVE_DATA_ROOT="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVE/data"
    DATASET_SPECIFIC_ARGS="--dataset ave --ave_data_root $AVE_DATA_ROOT"
    NUM_CLASSES=28
    
elif [ "$DATASET" = "avvp" ]; then
    # AVVP Dataset Configuration  
    AVVP_CSV_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_multimodal_simplified.csv"
    AVVP_AUDIO_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/audio/zip"
    AVVP_VIDEO_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/video/zip"
    DATASET_SPECIFIC_ARGS="--dataset avvp --avvp_csv_path $AVVP_CSV_PATH --avvp_audio_path $AVVP_AUDIO_PATH --avvp_video_path $AVVP_VIDEO_PATH"
    NUM_CLASSES=26  # 25 event classes + 1 background
fi

# =============================================================================
# ANALYSIS CONFIGURATION PARAMETERS
#
# These parameters control the depth and scope of the event-codebook analysis.
# Adjust them based on computational resources and desired analytical detail.
# =============================================================================

# Output directory with dataset and model type specification
OUTPUT_DIR="/project/ag-jafra/Souptik/CMG_New/Experiments/Misc/codebook_event_analysis/${DATASET}_${ANALYSIS_TYPE}_$(date +%Y%m%d_%H%M%S)"

# Sampling parameters for event class analysis
SAMPLES_PER_CLASS=3       # Number of samples per event class for balanced analysis
                         # More samples provide better statistical validity but increase runtime

# Model architecture parameters - must match the training configuration exactly
if [ "$MODEL_TYPE" = "AVT" ]; then
    # AVT Model Architecture
    AUDIO_DIM=128            # Audio feature dimension for VGGSound-style models
    VIDEO_DIM=512            # Video feature dimension for VGGSound-style models  
    TEXT_DIM=256             # Text feature dimension for AVT models
    VIDEO_OUTPUT_DIM=2048    # Video spatial output dimension after conv processing
    N_EMBEDDINGS=400         # Codebook size for event-aware quantization
    EMBEDDING_DIM=256        # Dimension of semantic representations
else
    # AV Model Architecture  
    AUDIO_DIM=128
    VIDEO_DIM=512
    TEXT_DIM=256             # Still need to specify even if not used
    VIDEO_OUTPUT_DIM=2048
    N_EMBEDDINGS=400
    EMBEDDING_DIM=256
fi

echo "Starting comprehensive event-codebook analysis..."
echo "Configuration Summary:"
echo "  Dataset: $DATASET (${NUM_CLASSES} classes)"
echo "  Model type: $MODEL_TYPE"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Analysis type: $ANALYSIS_TYPE"  
echo "  Output directory: $OUTPUT_DIR"
echo "  Samples per event class: $SAMPLES_PER_CLASS"
echo "  Model architecture: Audio=${AUDIO_DIM}D, Video=${VIDEO_DIM}D, Text=${TEXT_DIM}D"
echo "  Codebook configuration: ${N_EMBEDDINGS} vectors of ${EMBEDDING_DIM} dimensions"
echo "  Video processing: ${VIDEO_DIM}D -> ${VIDEO_OUTPUT_DIM}D spatial features"

# =============================================================================
# PRE-EXECUTION VALIDATION
#
# Comprehensive checks to prevent wasted computational time by verifying
# all prerequisites before starting the intensive analysis process.
# =============================================================================

# Verify that the specified checkpoint file exists and is accessible
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "==========================================="
    echo "ERROR: Model checkpoint file not found!"
    echo "Expected location: $CHECKPOINT_PATH"
    echo ""
    echo "Please verify:"
    echo "1. The checkpoint path is correct for your model type ($MODEL_TYPE)"
    echo "2. The model was successfully saved during training"
    echo "3. File system permissions allow access"
    echo "4. The model type matches the checkpoint (AV vs AVT)"
    echo "==========================================="
    exit 1
fi

# Verify dataset paths exist
if [ "$DATASET" = "ave" ] && [ ! -d "$AVE_DATA_ROOT" ]; then
    echo "==========================================="
    echo "ERROR: AVE dataset directory not found!"
    echo "Expected location: $AVE_DATA_ROOT"
    echo "Please verify the AVE data has been properly extracted and organized."
    echo "==========================================="
    exit 1
elif [ "$DATASET" = "avvp" ]; then
    if [ ! -f "$AVVP_CSV_PATH" ]; then
        echo "ERROR: AVVP CSV file not found at $AVVP_CSV_PATH"
        exit 1
    fi
    if [ ! -d "$AVVP_AUDIO_PATH" ] || [ ! -d "$AVVP_VIDEO_PATH" ]; then
        echo "ERROR: AVVP feature directories not found"
        echo "Audio path: $AVVP_AUDIO_PATH"
        echo "Video path: $AVVP_VIDEO_PATH"
        exit 1
    fi
fi

# Verify that the Python analysis script exists
PYTHON_SCRIPT_PATH="$SRCDIR/CMG_trial1/src/codebook_analysis_event.py"
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "==========================================="
    echo "ERROR: Python event analysis script not found!"
    echo "Expected location: $PYTHON_SCRIPT_PATH"
    echo ""
    echo "Please ensure the codebook_analysis_event.py script has been saved"
    echo "to the correct directory with proper permissions."
    echo "==========================================="
    exit 1
fi

echo "All prerequisites verified successfully."
echo "Checkpoint file confirmed: $(ls -lh $CHECKPOINT_PATH)"
echo "Python script confirmed: $(ls -lh $PYTHON_SCRIPT_PATH)"
if [ "$DATASET" = "ave" ]; then
    echo "AVE dataset confirmed: $(ls -d $AVE_DATA_ROOT)"
elif [ "$DATASET" = "avvp" ]; then
    echo "AVVP CSV confirmed: $(ls -lh $AVVP_CSV_PATH)"
    echo "AVVP features confirmed: Audio($(ls $AVVP_AUDIO_PATH | wc -l) files), Video($(ls $AVVP_VIDEO_PATH | wc -l) files)"
fi

# Navigate to the source directory for proper module imports
cd $SRCDIR/CMG_trial1/src

# =============================================================================
# MAIN ANALYSIS EXECUTION
#
# Launch the comprehensive event-codebook analysis with all specified parameters.
# This process will analyze how different event classes utilize the learned
# codebook representations.
# =============================================================================

echo ""
echo "Launching comprehensive event-codebook analysis..."
echo "This process will:"
echo "1. Sample representative data from each event class"
echo "2. Extract quantized representations for audio and video modalities"
if [ "$MODEL_TYPE" = "AVT" ]; then
    echo "3. Analyze cross-modal quantization patterns (Audio-Video-Text model)"
else
    echo "3. Analyze cross-modal quantization patterns (Audio-Video model)"
fi
echo "4. Generate event class usage heatmaps and diversity analysis"
echo "5. Create comprehensive visualizations of codebook specialization"
echo "6. Identify event-specific quantization patterns and cross-modal consistency"
echo ""

python codebook_analysis_event.py \
    $DATASET_SPECIFIC_ARGS \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_type $MODEL_TYPE \
    --samples_per_class $SAMPLES_PER_CLASS \
    --audio_dim $AUDIO_DIM \
    --video_dim $VIDEO_DIM \
    --text_dim $TEXT_DIM \
    --video_output_dim $VIDEO_OUTPUT_DIM \
    --n_embeddings $N_EMBEDDINGS \
    --embedding_dim $EMBEDDING_DIM

# =============================================================================
# POST-EXECUTION ANALYSIS AND REPORTING
#
# Provide immediate feedback about completion status and guide toward next steps
# in the research workflow for understanding event-based quantization patterns.
# =============================================================================

# Capture exit code to determine analysis success
ANALYSIS_EXIT_CODE=$?

echo ""
echo "==========================================="

if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Event-codebook analysis completed successfully!"
    echo ""
    echo "Analysis Results Available:"
    echo "  Primary output directory: $OUTPUT_DIR"
    echo "  Visualizations: $OUTPUT_DIR/visualizations/"
    echo "  Event class heatmaps: *_event_codebook_heatmaps.png"
    echo "  Top indices analysis: *_event_top_indices.png"
    echo "  Diversity analysis: event_diversity_analysis.png"
    echo ""
    echo "Key Research Questions Answered:"
    echo "1. Event Specialization: Which codebook regions specialize for specific events?"
    echo "2. Cross-Modal Consistency: Do audio/video use similar patterns for same events?"
    if [ "$DATASET" = "avvp" ]; then
        echo "3. Temporal Patterns: How do quantization patterns change across time segments?"
    fi
    echo "4. Model Efficiency: Which codebook vectors are most/least utilized?"
    echo ""
    echo "Recommended Next Steps:"
    echo "1. Examine event class heatmaps for specialization patterns"
    echo "2. Compare audio vs video quantization consistency"
    echo "3. Identify event classes with unique codebook signatures"
    echo "4. Analyze diversity metrics to understand model representation efficiency"
    if [ "$MODEL_TYPE" = "AVT" ]; then
        echo "5. Compare AVT quantization patterns with AV-only models"
    fi
    echo ""
    if [ -d "$OUTPUT_DIR/visualizations" ]; then
        echo "Visualization files created:"
        echo "$(find $OUTPUT_DIR/visualizations -name '*.png' | wc -l) visualization files generated"
        echo "Total output size: $(du -sh $OUTPUT_DIR | cut -f1)"
    fi
    
else
    echo "ERROR: Event analysis failed with exit code $ANALYSIS_EXIT_CODE"
    echo ""
    echo "Troubleshooting Steps:"
    echo "1. Check SLURM error log for detailed error messages"
    echo "2. Verify model checkpoint compatibility with specified architecture"
    echo "3. Ensure dataset paths are correct and accessible"
    echo "4. Confirm sufficient GPU memory for the model and analysis"
    echo ""
    echo "Common Issues for Event Analysis:"
    echo "- Model type mismatch (AV vs AVT) with checkpoint"
    echo "- Dataset format incompatibility (HDF5 for AVE, ZIP/pickle for AVVP)"
    echo "- Missing or corrupted feature files"
    echo "- Insufficient memory for large codebook analysis"
    echo "- Python dependency issues (h5py, zipfile, pickle)"
    echo ""
    echo "Dataset-Specific Issues:"
    if [ "$DATASET" = "ave" ]; then
        echo "- AVE: Check HDF5 file integrity and structure"
        echo "- AVE: Verify train/test split files exist"
    else
        echo "- AVVP: Check ZIP file accessibility and pickle format compatibility"  
        echo "- AVVP: Verify CSV structure and event label parsing"
    fi
fi

echo "Job completion time: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Dataset analyzed: $DATASET ($NUM_CLASSES event classes)"
echo "Model architecture: $MODEL_TYPE"
echo "==========================================="

# Exit with the same code as the Python script
exit $ANALYSIS_EXIT_CODE
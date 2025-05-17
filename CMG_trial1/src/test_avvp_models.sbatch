#!/bin/bash

#SBATCH --job-name=AVVP_test
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu007
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=24:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVVP_Test_Eval_2/test_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVVP_Test_Eval_2/test_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/AVVP_Test_Eval_2

# Set result output directories - separate for each model
OUTPUT_DIR_A2V=$SRCDIR/CMG_trial1/Results/AVVP_Test_Eval_2/A2V
OUTPUT_DIR_V2A=$SRCDIR/CMG_trial1/Results/AVVP_Test_Eval_2/V2A

# Create output directories
mkdir -p $OUTPUT_DIR_A2V/visualizations
mkdir -p $OUTPUT_DIR_A2V/statistics
mkdir -p $OUTPUT_DIR_V2A/visualizations
mkdir -p $OUTPUT_DIR_V2A/statistics

# Model paths
A2V_MODEL_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVVP_Test/A2V/final_model/AVT_model_epoch_49.pt"
V2A_MODEL_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVVP_Test/V2A/final_model/AVT_model_epoch_49.pt"

# Dataset paths - based on downstream training
A2V_CSV_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_eval_visual_checked_combined.csv"
V2A_CSV_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/data/AVVP_eval_audio_checked_combined.csv"
AUDIO_FEA_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/audio/zip"
VIDEO_FEA_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVVP/feature/video/zip"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Print environment info for debugging
echo "Python: $(which python)"
echo "GPU available: $(nvidia-smi)"
echo "Testing started at: $(date)"

# Save the dataset and testing scripts to the src directory
# mkdir -p $SRCDIR/CMG_trial1/src


# Run the test script for A2V model
echo "Testing A2V model..."
python $SRCDIR/CMG_trial1/src/test_avvp_models.py \
    --gpu 0 \
    --batch_size 64 \
    --output_dir "$OUTPUT_DIR_A2V" \
    --model_path "$A2V_MODEL_PATH" \
    --model_type "A2V" \
    --meta_csv_path "$A2V_CSV_PATH" \
    --fea_path "$VIDEO_FEA_PATH"

# Run the test script for V2A model
echo "Testing V2A model..."
python $SRCDIR/CMG_trial1/src/test_avvp_models.py \
    --gpu 0 \
    --batch_size 64 \
    --output_dir "$OUTPUT_DIR_V2A" \
    --model_path "$V2A_MODEL_PATH" \
    --model_type "V2A" \
    --meta_csv_path "$V2A_CSV_PATH" \
    --fea_path "$AUDIO_FEA_PATH"

# Create a combined summary comparing both models
echo "Creating combined summary..."
COMBINED_DIR=$SRCDIR/CMG_trial1/Results/AVVP_Test_Eval_2/combined
mkdir -p $COMBINED_DIR

# Extract metrics from individual summaries
if [ -f "$OUTPUT_DIR_A2V/summary.json" ] && [ -f "$OUTPUT_DIR_V2A/summary.json" ]; then
    A2V_PRECISION=$(grep "precision" $OUTPUT_DIR_A2V/summary.json | head -1 | awk '{print $2}' | tr -d ',')
    A2V_RECALL=$(grep "recall" $OUTPUT_DIR_A2V/summary.json | head -1 | awk '{print $2}' | tr -d ',')
    A2V_F1=$(grep "f1_score" $OUTPUT_DIR_A2V/summary.json | head -1 | awk '{print $2}' | tr -d ',')

    V2A_PRECISION=$(grep "precision" $OUTPUT_DIR_V2A/summary.json | head -1 | awk '{print $2}' | tr -d ',')
    V2A_RECALL=$(grep "recall" $OUTPUT_DIR_V2A/summary.json | head -1 | awk '{print $2}' | tr -d ',')
    V2A_F1=$(grep "f1_score" $OUTPUT_DIR_V2A/summary.json | head -1 | awk '{print $2}' | tr -d ',')

    # Create combined summary using Python for calculations
    python -c "
import json
a2v_precision = float('${A2V_PRECISION}' or 0)
a2v_recall = float('${A2V_RECALL}' or 0)
a2v_f1 = float('${A2V_F1}' or 0)
v2a_precision = float('${V2A_PRECISION}' or 0)
v2a_recall = float('${V2A_RECALL}' or 0)
v2a_f1 = float('${V2A_F1}' or 0)

summary = {
    'A2V_precision': a2v_precision,
    'A2V_recall': a2v_recall,
    'A2V_f1_score': a2v_f1,
    'V2A_precision': v2a_precision,
    'V2A_recall': v2a_recall,
    'V2A_f1_score': v2a_f1,
    'precision_difference': abs(a2v_precision - v2a_precision),
    'recall_difference': abs(a2v_recall - v2a_recall),
    'f1_difference': abs(a2v_f1 - v2a_f1)
}

with open('${COMBINED_DIR}/combined_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
    " 
    echo "Combined summary created successfully."
else
    echo "ERROR: Summary files not found. Combined summary not created."
fi

echo "Testing complete at: $(date)"
echo "A2V results saved to $OUTPUT_DIR_A2V"
echo "V2A results saved to $OUTPUT_DIR_V2A"
echo "Combined summary saved to $COMBINED_DIR"
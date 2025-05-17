#!/bin/bash

#SBATCH --job-name=AVE_test
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu007
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=24:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVE_Test_Eval_2/test_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/AVE_Test_Eval_2/test_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories for logs
mkdir -p $SRCDIR/CMG_trial1/slurm/AVE_Test_Eval_2

# Set result output directories - separate for each model
OUTPUT_DIR_A2V=$SRCDIR/CMG_trial1/Results/AVE_Test_Eval_2/A2V
OUTPUT_DIR_V2A=$SRCDIR/CMG_trial1/Results/AVE_Test_Eval_2/V2A

# Create output directories
mkdir -p $OUTPUT_DIR_A2V/visualizations
mkdir -p $OUTPUT_DIR_A2V/statistics
mkdir -p $OUTPUT_DIR_V2A/visualizations
mkdir -p $OUTPUT_DIR_V2A/statistics

# Model paths
A2V_MODEL_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVE_Test/A2V/final_model/AVT_model_epoch_29.pt"
V2A_MODEL_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Models/AVE_Test/V2A/final_model/AVT_model_epoch_29.pt"

# Dataset paths
DATA_ROOT="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVE/data"
ANNOTATIONS_PATH="/project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/CMG/data/data/AVE/data/testSet.txt"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Print environment info for debugging
echo "Python: $(which python)"
echo "GPU available: $(nvidia-smi)"
echo "Testing started at: $(date)"

# Run the test script for A2V model
echo "Testing A2V model..."
python $SRCDIR/CMG_trial1/src/test_ave_models.py \
    --gpu 0 \
    --batch_size 64 \
    --output_dir "$OUTPUT_DIR_A2V" \
    --model_path "$A2V_MODEL_PATH" \
    --model_type "A2V" \
    --data_root "$DATA_ROOT" \
    --annotations_path "$ANNOTATIONS_PATH"

# Run the test script for V2A model
echo "Testing V2A model..."
python $SRCDIR/CMG_trial1/src/test_ave_models.py \
    --gpu 0 \
    --batch_size 64 \
    --output_dir "$OUTPUT_DIR_V2A" \
    --model_path "$V2A_MODEL_PATH" \
    --model_type "V2A" \
    --data_root "$DATA_ROOT" \
    --annotations_path "$ANNOTATIONS_PATH"

# Create a combined summary comparing both models
echo "Creating combined summary..."
COMBINED_DIR=$SRCDIR/CMG_trial1/Results/AVE_Test_Eval_2/combined
mkdir -p $COMBINED_DIR

# Extract accuracies from individual summaries
A2V_ACC=$(grep "accuracy" $OUTPUT_DIR_A2V/summary.json | awk '{print $2}' | tr -d ',')
V2A_ACC=$(grep "accuracy" $OUTPUT_DIR_V2A/summary.json | awk '{print $2}' | tr -d ',')

# Create combined summary
cat > $COMBINED_DIR/combined_summary.json << EOL
{
  "A2V_accuracy": $A2V_ACC,
  "V2A_accuracy": $V2A_ACC,
  "accuracy_difference": $(echo "$A2V_ACC - $V2A_ACC" | bc | awk '{print sqrt($1*$1)}')
}
EOL

echo "Testing complete at: $(date)"
echo "A2V results saved to $OUTPUT_DIR_A2V"
echo "V2A results saved to $OUTPUT_DIR_V2A"
echo "Combined summary saved to $COMBINED_DIR"
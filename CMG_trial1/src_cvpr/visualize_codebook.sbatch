#!/bin/bash

#SBATCH --job-name=viz_codebook_final
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=02:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/visualizations/viz_final_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/visualizations/viz_final_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

# Create output directories
mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/visualizations
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Visualizations

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Set checkpoint paths - ADJUST THESE TO YOUR ACTUAL PATHS
NOVEL_CHECKPOINT="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Ablation_pretrain/AVT/400Codebook/checkpoint/Ablation-AVT-5.pt"
DCID_CHECKPOINT="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/DCID/40k/AVT_vgg40k_size400.pt"
MICU_CHECKPOINT="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/MICU/40k/checkpoint/MICU-step2400.pt"

OUTPUT_DIR="$SRCDIR/CMG_trial1/Benchmarks/Visualizations"

# Model configuration parameters (from your pretraining setup)
AUDIO_DIM=128
VIDEO_DIM=512
TEXT_DIM=256
N_EMBEDDINGS=400
EMBEDDING_DIM=256  # Per segment, total will be 768 (256*3) for novel model

echo "=========================================="
echo "Codebook Visualization with Model Loading"
echo "=========================================="
echo ""
echo "Model Configuration:"
echo "  Audio dim: $AUDIO_DIM"
echo "  Video dim: $VIDEO_DIM"
echo "  Text dim: $TEXT_DIM"
echo "  Number of embeddings: $N_EMBEDDINGS"
echo "  Embedding dim per segment: $EMBEDDING_DIM"
echo "  Total embedding dim (Novel): $((EMBEDDING_DIM * 3))"
echo ""

# Navigate to project directory
cd $SRCDIR/CMG_trial1

# Visualize Novel Model
echo "=========================================="
echo "Processing Novel Model (Hierarchical AVT)"
echo "=========================================="
python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/visualize_codebooks.py \
    --checkpoint_path "$NOVEL_CHECKPOINT" \
    --model_type novel \
    --model_name "Novel" \
    --output_dir "$OUTPUT_DIR" \
    --audio_dim $AUDIO_DIM \
    --video_dim $VIDEO_DIM \
    --text_dim $TEXT_DIM \
    --n_embeddings $N_EMBEDDINGS \
    --embedding_dim $EMBEDDING_DIM \
    --save_embeddings "$OUTPUT_DIR/novel_tsne_embeddings.npy"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Novel model visualization complete!"
    echo ""
else
    echo ""
    echo "✗ Novel model visualization failed!"
    echo "Check error log for details"
    echo ""
fi

# Visualize DCID Model
echo "=========================================="
echo "Processing DCID Model"
echo "=========================================="
python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/visualize_codebooks.py \
    --checkpoint_path "$DCID_CHECKPOINT" \
    --model_type dcid \
    --model_name "DCID" \
    --output_dir "$OUTPUT_DIR" \
    --audio_dim $AUDIO_DIM \
    --video_dim $VIDEO_DIM \
    --text_dim $TEXT_DIM \
    --n_embeddings $N_EMBEDDINGS \
    --embedding_dim $EMBEDDING_DIM \
    --save_embeddings "$OUTPUT_DIR/dcid_tsne_embeddings.npy"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ DCID model visualization complete!"
    echo ""
else
    echo ""
    echo "✗ DCID model visualization failed!"
    echo "Check error log for details"
    echo ""
fi

# Visualize MICU Model
echo "=========================================="
echo "Processing MICU Model"
echo "=========================================="
python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/visualize_codebooks.py \
    --checkpoint_path "$MICU_CHECKPOINT" \
    --model_type micu \
    --model_name "MICU" \
    --output_dir "$OUTPUT_DIR" \
    --audio_dim $AUDIO_DIM \
    --video_dim $VIDEO_DIM \
    --text_dim $TEXT_DIM \
    --n_embeddings $N_EMBEDDINGS \
    --embedding_dim $EMBEDDING_DIM \
    --save_embeddings "$OUTPUT_DIR/micu_tsne_embeddings.npy"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ MICU model visualization complete!"
    echo ""
else
    echo ""
    echo "✗ MICU model visualization failed!"
    echo "Check error log for details"
    echo ""
fi

echo "=========================================="
echo "All Visualizations Completed!"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh $OUTPUT_DIR/*.png $OUTPUT_DIR/*.npy 2>/dev/null || echo "  No files found - check error logs"
echo ""
echo "To view results:"
echo "  cd $OUTPUT_DIR"
echo "  ls -lh"
echo ""
echo "Log files:"
echo "  Output: $SRCDIR/CMG_trial1/slurm_cvpr/visualizations/viz_final_${SLURM_JOB_ID}.out"
echo "  Errors: $SRCDIR/CMG_trial1/slurm_cvpr/visualizations/viz_final_${SLURM_JOB_ID}.err"
echo "=========================================="
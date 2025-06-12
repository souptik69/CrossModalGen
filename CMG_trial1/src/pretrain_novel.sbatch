#!/bin/bash

#SBATCH --job-name=AV_pretrain_final
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu005
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=65:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/5_Finalmodel_40k_Novel_AV/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/5_Finalmodel_40k_Novel_AV/pretrain_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/5_Finalmodel_40k_Novel_AV

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run pretraining
python $SRCDIR/CMG_trial1/src/pretrain_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Novel_Model_Final/Equal_Hier_AV/40k/" \
    --n_epoch 6 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "vggsound" \
    --model_save_path "$SRCDIR/CMG_trial1/Novel_Model_Final/Equal_Hier_AV/40k/checkpoint/" \
    --print_freq 1

echo "Pretraining complete!"
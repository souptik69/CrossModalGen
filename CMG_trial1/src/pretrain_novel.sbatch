#!/bin/bash

#SBATCH --job-name=AV_pretrain_MetaHierSigmoid
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu006
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=165:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/1_Novel_AV_MetaHierSigmoid/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/1_Novel_AV_MetaHierSigmoid/pretrain_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/1_Novel_AV_MetaHierSigmoid

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run pretraining
python $SRCDIR/CMG_trial1/src/pretrain_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Models/Novel_AV_final_3/MetaHierSigmoid/" \
    --n_epoch 6 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "vggsound" \
    --print_freq 1

echo "Pretraining complete!"
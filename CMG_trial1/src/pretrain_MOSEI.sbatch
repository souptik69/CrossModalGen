#!/bin/bash

#SBATCH --job-name=MOSEI_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=65:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/2_NoLSTM_Final_50_reg_MOSEI_supervised_TAV/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/2_NoLSTM_Final_50_reg_MOSEI_supervised_TAV/pretrain_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/2_NoLSTM_Final_50_reg_MOSEI_supervised_TAV

mkdir -p $SRCDIR/CMG_trial1/MOSEI_Models2/2_NoLSTM_seq50_50_supervised_reg_TAV_multimodal/checkpoint


source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run pretraining
python $SRCDIR/CMG_trial1/src/pretrain_MOSEI_supervised_combined.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Models2/2_NoLSTM_seq50_50_supervised_reg_TAV_multimodal/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Models2/2_NoLSTM_seq50_50_supervised_reg_TAV_multimodal/checkpoint/" \
    --print_freq 1 \
    --loss_csv_path "training_losses.csv" \
    --val_loss_csv_path "val_losses.csv"

echo "Pretraining complete!"
#!/bin/bash

#SBATCH --job-name=MOSEI_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=65:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/1_NoLSTM_Final_50_reg_MOSEI_unsupervised_TAV/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/1_NoLSTM_Final_50_reg_MOSEI_unsupervised_TAV/pretrain_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/1_NoLSTM_Final_50_reg_MOSEI_unsupervised_TAV

mkdir -p $SRCDIR/CMG_trial1/MOSEI_Models2/2_LSTM_seq50_50_supervised_reg_TAV_L1/checkpoint_1


source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run pretraining
python $SRCDIR/CMG_trial1/src/pretrain_MOSEI_supervised_modal.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Models2/2_LSTM_seq50_50_supervised_reg_TAV_L1/" \
    --n_epoch 100 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Models2/2_LSTM_seq50_50_supervised_reg_TAV_L1/checkpoint_1/" \
    --print_freq 1 \
    --loss_csv_path "training_losses_1.csv" \
    --val_loss_csv_path "val_losses_1.csv"

echo "Pretraining complete!"
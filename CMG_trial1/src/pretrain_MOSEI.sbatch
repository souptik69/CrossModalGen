#!/bin/bash

#SBATCH --job-name=MOSEI_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=65:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/LSTM_2_100_reg_MOSEI_supervised_VAT/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/LSTM_2_100_reg_MOSEI_supervised_VAT/pretrain_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/LSTM_2_100_reg_MOSEI_supervised_VAT

mkdir -p $SRCDIR/CMG_trial1/MOSEI_Models2/LSTM_seq50_100_supervised_reg_VAT/checkpoint


source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run pretraining
python $SRCDIR/CMG_trial1/src/pretrain_MOSEI_supervised.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Models2/LSTM_seq50_100_supervised_reg_VAT/" \
    --n_epoch 100 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Models2/LSTM_seq50_100_supervised_reg_VAT/checkpoint/" \
    --print_freq 1 \
    --loss_csv_path "training_losses.csv" \
    --val_loss_csv_path "val_losses.csv"

echo "Pretraining complete!"
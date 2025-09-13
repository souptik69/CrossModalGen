#!/bin/bash

#SBATCH --job-name=MOSEI_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=65:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/256_NoLSTM_Final_10_reg_MOSEI_unsupervised_VAT_L2/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/256_NoLSTM_Final_10_reg_MOSEI_unsupervised_VAT_L2/pretrain_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/256_NoLSTM_Final_10_reg_MOSEI_unsupervised_VAT_L2

mkdir -p $SRCDIR/CMG_trial1/MOSEI_Models2/256codebook_Norm_seq10_10_unsupervised_reg_TAV_L2_test/checkpoint_1


source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run pretraining
python $SRCDIR/CMG_trial1/src/pretrain_MOSEI_unsupervised.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/MOSEI_Models2/256codebook_Norm_seq10_10_unsupervised_reg_TAV_L2_test/" \
    --n_epoch 15 \
    --batch_size 16 \
    --test_batch_size 16 \
    --dataset_name "mosei" \
    --model_save_path "$SRCDIR/CMG_trial1/MOSEI_Models2/256codebook_Norm_seq10_10_unsupervised_reg_TAV_L2_test/checkpoint_1/" \
    --print_freq 1 \
    --loss_csv_path "training_losses_1.csv" \
    --val_loss_csv_path "val_losses_1.csv"

echo "Pretraining complete!"
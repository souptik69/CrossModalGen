#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Downstream_Benchmark_AVT/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Downstream_Benchmark_AVT/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/Downstream_Benchmark_AVT
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVVP/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE2AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE2AVVP/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/UCF2CGGSOUND/vgga_ucfv
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/UCF2CGGSOUND/ucfv_vgga


# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/CMG_final

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE/A2V/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/ave.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE/V2A/" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1


python $SRCDIR/CMG_trial1/src_cvpr/avvp.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1


echo "AVVP training complete!"


# Run AVE to AVVP training

python $SRCDIR/CMG_trial1/src_cvpr/ave_avvp.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE2AVVP/V2A/" \
    --n_epoch 30 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ave_avvp_va" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1

python $SRCDIR/CMG_trial1/src_cvpr/ave_avvp.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/AVE2AVVP/A2V/" \
    --n_epoch 30 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ave_avvp_av" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1

echo "AVE to AVVP training complete!"

# Run UCF to VGGSOUND training

python $SRCDIR/CMG_trial1/src_cvpr/ucf_vggsound.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/UCF2CGGSOUND/vgga_ucfv/" \
    --n_epoch 30 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vgga_ucfv" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1

python $SRCDIR/CMG_trial1/src_cvpr/ucf_vggsound.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream/AVT/40k/UCF2CGGSOUND/ucfv_vgga/" \
    --n_epoch 30 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ucfv_vgga" \
    --modality "AVT" \
    --print_freq 1 \
    --eval_freq 1

echo "UCF to VGGSOUND training complete!"
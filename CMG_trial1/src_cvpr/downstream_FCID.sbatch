#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Downstream_Benchmark_FCID/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Downstream_Benchmark_FCID/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

CHECKPOINT_PATH_AVT_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/FCID/40k/FCID.pt"
CHECKPOINT_PATH_AVT_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/FCID/90k/checkpoint/FCID-step5400.pt"


mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVVP/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE2AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE2AVVP/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/UCF2CGGSOUND/vgga_ucfv
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/UCF2CGGSOUND/ucfv_vgga



# Run AVE training

# python $SRCDIR/CMG_trial1/src_cvpr/ave_FCID.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE/A2V/" \
#     --n_epoch 25 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_av" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
#     --print_freq 1 \
#     --eval_freq 1 \
#     --toc_max_num 1 \
#     --toc_min_num 0 \
#     --choose_channel 256


# python $SRCDIR/CMG_trial1/src_cvpr/ave_FCID.py \
#     --gpu 0 \
#     --lr 0.00025 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE/V2A/" \
#     --n_epoch 25 \
#     --batch_size 256 \
#     --test_batch_size 128 \
#     --dataset_name "ave_va" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
#     --print_freq 1 \
#     --eval_freq 1 \
#     --toc_max_num 1 \
#     --toc_min_num 0 \
#     --choose_channel 256


# echo "AVE training complete!"


# # Run AVVP training
# python $SRCDIR/CMG_trial1/src_cvpr/avvp_FCID.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_va" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
#     --print_freq 1 \
#     --eval_freq 1 \
#     --toc_max_num 1 \
#     --toc_min_num 0 \
#     --choose_channel 196


# python $SRCDIR/CMG_trial1/src_cvpr/avvp_FCID.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 64 \
#     --test_batch_size 64 \
#     --dataset_name "avvp_av" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
#     --print_freq 1 \
#     --eval_freq 1 \
#     --toc_max_num 1 \
#     --toc_min_num 0 \
#     --choose_channel 196


# echo "AVVP training complete!"


# # Run AVE to AVVP training

# python $SRCDIR/CMG_trial1/src_cvpr/ave_avvp_FCID.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE2AVVP/V2A/" \
#     --n_epoch 50 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "ave_avvp_va" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
#     --print_freq 1 \
#     --eval_freq 1 \
#     --toc_max_num 1 \
#     --toc_min_num 0 \
#     --choose_channel 128

# python $SRCDIR/CMG_trial1/src_cvpr/ave_avvp_FCID.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/AVE2AVVP/A2V/" \
#     --n_epoch 50 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "ave_avvp_av" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
#     --print_freq 1 \
#     --eval_freq 1 \
#     --toc_max_num 1 \
#     --toc_min_num 0 \
#     --choose_channel 128

# echo "AVE to AVVP training complete!"

# # Run UCF to VGGSOUND training

# python $SRCDIR/CMG_trial1/src_cvpr/ucf_vggsound_FCID.py \
#     --gpu 0 \
#     --lr 0.0004 \
#     --clip_gradient 0.5 \
#     --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/UCF2CGGSOUND/vgga_ucfv/" \
#     --n_epoch 80 \
#     --batch_size 80 \
#     --test_batch_size 64 \
#     --dataset_name "vgga_ucfv" \
#     --modality "AVT" \
#     --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
#     --print_freq 1 \
#     --eval_freq 1 \
#     --toc_max_num 1 \
#     --toc_min_num 0 \
#     --choose_channel 168

python $SRCDIR/CMG_trial1/src_cvpr/ucf_vggsound_FCID.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/40k/UCF2CGGSOUND/ucfv_vgga/" \
    --n_epoch 80 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ucfv_vgga" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 168

echo "UCF to VGGSOUND training complete!"





mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVVP/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE2AVVP/A2V
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE2AVVP/V2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/UCF2CGGSOUND/vgga_ucfv
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/UCF2CGGSOUND/ucfv_vgga

# Run AVE training
python $SRCDIR/CMG_trial1/src_cvpr/ave_FCID.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE/A2V/" \
    --n_epoch 25 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 256


python $SRCDIR/CMG_trial1/src_cvpr/ave_FCID.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE/V2A/" \
    --n_epoch 25 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_va" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 256


echo "AVE training complete!"


# Run AVVP training
python $SRCDIR/CMG_trial1/src_cvpr/avvp_FCID.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_va" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 196


python $SRCDIR/CMG_trial1/src_cvpr/avvp_FCID.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 196


echo "AVVP training complete!"


# Run AVE to AVVP training

python $SRCDIR/CMG_trial1/src_cvpr/ave_avvp_FCID.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE2AVVP/V2A/" \
    --n_epoch 50 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ave_avvp_va" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 128

python $SRCDIR/CMG_trial1/src_cvpr/ave_avvp_FCID.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/AVE2AVVP/A2V/" \
    --n_epoch 50 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ave_avvp_av" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 128

echo "AVE to AVVP training complete!"

# Run UCF to VGGSOUND training

python $SRCDIR/CMG_trial1/src_cvpr/ucf_vggsound_FCID.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/UCF2CGGSOUND/vgga_ucfv/" \
    --n_epoch 80 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vgga_ucfv" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 168

python $SRCDIR/CMG_trial1/src_cvpr/ucf_vggsound_FCID.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/CMG_trial1/Benchmarks/Downstream_FCID/AVT/90k/UCF2CGGSOUND/ucfv_vgga/" \
    --n_epoch 80 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ucfv_vgga" \
    --modality "AVT" \
    --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
    --print_freq 1 \
    --eval_freq 1 \
    --toc_max_num 1 \
    --toc_min_num 0 \
    --choose_channel 168

echo "UCF to VGGSOUND training complete!"
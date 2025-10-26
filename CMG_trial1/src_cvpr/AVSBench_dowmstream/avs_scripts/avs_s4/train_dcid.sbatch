#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVS_Results_Benchmark/pretrain_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVS_Results_Benchmark/pretrain_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/AVS_Results_Benchmark
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/40k/train/T2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/40k/train/A2T
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/90k/train/T2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/90k/train/A2T

CHECKPOINT_PATH_AVT_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/DCID/40k/AVT_vgg40k_size400.pt"
CHECKPOINT_PATH_AVT_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Models/AVT/DCID/90k/checkpoint/DCID-model-4.pt"

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/CMG_final

setting='S4'
visual_backbone="pvt"

#Train T2A 40k

python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/train_ta.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --max_epoches 4 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/40k/train/T2A/" \
        --tpavi_va_flag 


#Train A2T 40k

python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/train_at.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --max_epoches 4 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/40k/train/A2T/" \
        --tpavi_va_flag 


#Train T2A 90k

python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/train_ta.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --max_epoches 4 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/90k/train/T2A/" \
        --tpavi_va_flag 


#Train A2T 90k

python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/train_at.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --max_epoches 4 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_DCID_ep4/90k/train/A2T/" \
        --tpavi_va_flag 

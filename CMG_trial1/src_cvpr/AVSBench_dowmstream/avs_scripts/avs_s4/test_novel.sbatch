#!/bin/bash

#SBATCH --job-name=AVT_pretest
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVS_Results_Novel_test/pretest_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/AVS_Results_Novel_test/pretest_%j.err

# Set source directory
SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm_cvpr/AVS_Results_Novel_test
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/40k/test/T2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/40k/test/A2T
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/90k/test/T2A
mkdir -p $SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/90k/test/A2T

CHECKPOINT_PATH_AVT_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AVT/40k/checkpoint/Ablation-AVT-5.pt"
CHECKPOINT_PATH_AVT_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Novel_Models/AVT/90k/checkpoint/HierVQ-model-AVT-5.pt"

# Activate conda environment

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new


setting='S4'
visual_backbone="pvt"

#test T2A 40k
weight_TA_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/40k/train/T2A/S4_pvt_20251026-112923/S4_pvt_best.pth"

python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/test_ta_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --test_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --weights "$weight_TA_40k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/40k/test/T2A/" \
        --tpavi_va_flag \
        --save_pred_mask 


#test A2T 40k
weight_AT_40k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/40k/train/A2T/S4_pvt_20251026-120629/S4_pvt_best.pth"

python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/test_at_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --test_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --weights "$weight_AT_40k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/40k/test/A2T/" \
        --tpavi_va_flag \
        --save_pred_mask 


#test T2A 90k

weight_TA_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/90k/train/T2A/S4_pvt_20251026-123212/S4_pvt_best.pth"
python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/test_ta_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --test_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
        --weights "$weight_TA_90k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/90k/test/T2A/" \
        --tpavi_va_flag \
        --save_pred_mask 


#test A2T 90k

weight_AT_90k="/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/90k/train/A2T/S4_pvt_20251026-125756/S4_pvt_best.pth"
python $SRCDIR/CMG_trial1/src_cvpr/AVSBench_dowmstream/avs_scripts/avs_s4/test_at_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --test_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_90k" \
        --weights "$weight_AT_90k" \
        --log_dir "$SRCDIR/CMG_trial1/Benchmarks/Downstream_AVS_Novel_4ep/90k/test/A2T/" \
        --tpavi_va_flag \
        --save_pred_mask 

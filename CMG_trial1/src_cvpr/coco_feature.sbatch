#!/bin/bash

#SBATCH --job-name=mscoco_vgg19
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/COCO/mscoco_extract_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/COCO/mscoco_extract_%j.err

mkdir -p /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/COCO
mkdir -p /project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/val2017_features

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/VGGSoundAVEL/CMG/envs/visual_ex

echo "Python: $(which python)"
echo "Starting MSCOCO VGG19 feature extraction..."

python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/coco_feature.py \
    --image_dir /project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/val2017 \
    --annotations_json /project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/annotations/captions_val2017.json \
    --output_dir /project/ag-jafra/Souptik/VGGSoundAVEL/Data_CMG/val2017_features \
    --batch_size 64

if [ $? -eq 0 ]; then
    echo "SUCCESS! End Time: $(date)"
else
    echo "FAILED with exit code $?"
    exit 1
fi
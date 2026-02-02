#!/bin/bash

#SBATCH --job-name=clotho_vggish
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-10g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=10:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Clotho/clotho_extract_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Clotho/clotho_extract_%j.err

SRCDIR=/project/ag-jafra/Souptik/VGGSoundAVEL

mkdir -p /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm_cvpr/Clotho/
mkdir -p $SRCDIR/Data_CMG/clotho_audio_features

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

source ~/.bashrc
conda activate vggsound_env

echo "Python: $(which python)"
echo "Starting Clotho VGGish feature extraction..."

python /project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/src_cvpr/clotho_extractor.py \
    --audio_dir $SRCDIR/Data_CMG/evaluation \
    --csv_path $SRCDIR/Data_CMG/clotho_captions_evaluation.csv \
    --output_dir $SRCDIR/Data_CMG/clotho_audio_features \
    --checkpoint_path $SRCDIR/vggish_model/vggish_model.ckpt \
    --gpu_id 0 \
    --batch_size 16

if [ $? -eq 0 ]; then
    echo "SUCCESS! End Time: $(date)"
else
    echo "FAILED with exit code $?"
    exit 1
fi
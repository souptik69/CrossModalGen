#!/bin/bash

#SBATCH --job-name=Download_MOSEI
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu007
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=70:00:00
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Download/mosei_%j.out
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/CMG_trial1/slurm/0_Download/mosei_%j.err

SRCDIR=/project/ag-jafra/Souptik/CMG_New/Experiments

mkdir -p $SRCDIR/CMG_trial1/slurm/0_Download

source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Run download
python $SRCDIR/CMG_trial1/src/mosei_dataset/MOSEI_download.py

echo "MultiBench CMU-MOSEI download complete!"
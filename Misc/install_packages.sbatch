#!/bin/bash
#SBATCH --job-name=install_ml_packages
#SBATCH --output=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/env3/install_ml_packages_%j.log
#SBATCH --error=/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm//env3/install_ml_packages_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Define paths
ENV_PATH="/project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new"
SLURM_DIR="/project/ag-jafra/Souptik/CMG_New/Experiments/Slurm/env3"

# Create Slurm output directory if it doesn't exist
mkdir -p $SLURM_DIR



# # Activate the environment
# source activate $ENV_PATH

# Activate conda environment
source ~/.bashrc
conda activate /project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new

# Step 4: Install conda packages
# conda install -y mkl=2024.0 mkl-service=2.3.0 mkl_fft=1.2.0 mkl_random=1.1.1
# conda install -y blas=1.0 intel-openmp=2025.0.0
# conda install -y setuptools=65.6.3 wheel=0.38.4 pip=22.3.1

# echo "Installing PyTorch ecosystem..."
# # Install PyTorch with CUDA support (latest version)
# conda install -y pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# echo "Installing ML monitoring tools..."
# # Install TensorBoard and TensorBoardX
# conda install -y -c conda-forge tensorboard=2.15.2
# pip install tensorboardx==2.6.2.2

# echo "Installing MXNet and BERT-related packages..."
# # # Install MXNet
# pip install mxnet==1.9.1
# pip install bert-embedding==1.0.1 gluonnlp==0.6.0

# echo "Installing specialized PyTorch extensions..."
# # Install PyTorch Scatter and info-nce
# pip install torch-scatter==2.1.2
# pip install info-nce-pytorch==0.1.4

# echo "Installing additional ML utilities..."
# # Install additional packages with updated versions
# conda install -y -c conda-forge \
#     h5py=3.10.0 \
#     tqdm=4.66.2 \
#     pillow=10.2.0 \
    # scikit-learn=1.4.1 \
    # gensim=4.3.2
# conda install -c conda-forge libstdcxx-ng
# conda install -y -c conda-forge scikit-learn=1.4.1 gensim=4.3.2

# conda install -c conda-forge libstdcxx-ng scipy=1.12.0 scikit-learn=1.4.1 gensim=4.3.2

# echo "Installing development tools..."
# # Install development and utility packages
# conda install -y -c conda-forge \
#     ipython=8.19.0 \
#     notebook=7.0.6 \
#     matplotlib=3.8.2 \
#     pandas=2.1.4 \
#     numpy=1.26.3 \
#     scipy=1.12.0

# # For packages that might have compatibility issues
# echo "Checking for any package conflicts..."
# conda list

# echo "Creating environment file..."
# # Save environment configuration
# conda env export > $ENV_PATH/environment.yml
# conda list --explicit > $ENV_PATH/spec-file.txt

# echo "Testing PyTorch installation..."
# # Test PyTorch CUDA availability
# python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# pip install ruamel.yaml==0.18.1 ruamel.yaml-clib==0.2.8 \
#     requests==2.31.0 \
#     graphviz==0.20.1 \
#     typing==3.7.4.3 typing-extensions==4.12.0 \
#     six==1.16.0 python-dateutil==2.9.0.post0 pytz==2025.1 \
#     wrapt==1.16.0 \
#     smart-open==7.1.0 \
#     gensim==4.3.2 \
#     scikit-learn==1.4.1

# echo "Installation complete!"

# Add your environment's lib directory to LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new/lib

# Verify the LD_LIBRARY_PATH has been updated
# echo $LD_LIBRARY_PATH

# module load cuda/11.7
# export LD_LIBRARY_PATH=/project/ag-jafra/Souptik/CMG_New/Experiments/envs/CMG_new/lib:$LD_LIBRARY_PATH

# Now check if gensim imports correctly and get its version

# python -c "import gensim; print(f'Gensim import successful. Version: {gensim.__version__}')"

# python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"

# conda install -c conda-forge libstdcxx-ng

pip uninstall -y torch torchvision torchaudio
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

python -c "import torch, torchvision, torchaudio; print(f'PyTorch: {torch.__version__}, torchvision: {torchvision.__version__}, torchaudio: {torchaudio.__version__}')"
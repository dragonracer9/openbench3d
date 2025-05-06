#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=preprocessing

#SBATCH --chdir .
#SBATCH --account 3dv

#SBATCH --time=48:00:00
#SBATCH -o /home/%u/slurm_output__%x-%j.out

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1

set -e
set -o xtrace
echo PWD:$(pwd)
echo STARTING AT $(date)

# Environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate openmask3d

bash install_requirements.sh
pip install -e .
# ...

# Run your experiment
python -c "import torch; print('Cuda available?', torch.cuda.is_available())"
python -c "import torch; torch.manual_seed(72); print(torch.randn((3,3)))"

# cd class_agnostic_mask_computation
# python -m datasets.preprocessing.scannet_preprocessing preprocess --data_dir="PATH_TO_ORIGINAL_SCANNET_DATASET" --save_dir="data/processed/scannet" --git_repo="PATH_TO_SCANNET_GIT_REPO" --scannet200=true

echo "Done."
echo FINISHED at $(date)
#!/bin/sh
#SBATCH --job-name=JEPA
#SBATCH --account=csci_ga_2572-2024fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --open-mode=append
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

singularity exec --bind /scratch --nv \
  --overlay /scratch/qx690/overlay-25GB-500K.ext3:ro \ls
  /scratch/qx690/ubuntu-20.04.3.sif /bin/bash <<EOF
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate my_env
cd /scratch/qx690/DL24FA
python ./trainer.py
EOF

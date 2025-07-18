#!/bin/bash
#SBATCH --job-name=PSA_a2
#SBATCH --output=logs_run/psa_a2.o
#SBATCH --error=logs_run/psa_a2.e
#SBATCH --constraint=a100
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --account=xvy@a100
#SBATCH --export=ALL
module purge
module load arch/a100
module load pytorch-gpu/
wandb login

# Navigate to the directory containing your script

export WANDB_PROJECT="MaskMIFNO"
export WANDB_ENTITY="amine-grini-centralesup-lec"  # Optional if you're using teams
export WANDB_MODE=offline                   # or "offline", "dryrun"
export WANDB_DIR=$SCRATCH/MIFNO_logs
# Run the Python script
python models/train_lightning_fixed.py --batch_size 16 --learning_rate 4e-4

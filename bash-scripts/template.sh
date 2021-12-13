#!/bin/bash
#SBATCH -A snic2021-22-879
#SBATCH --get-user-env
#SBATCH --time=0-03:00:00
#SBATCH --cpus-per-task=1

date
pwd

git rev-parse --short HEAD

module load python/3.6.8
#module load python
#pip3 install --user -r ../concept-history-cluster/requirements.txt

export CONFIG={}

# Create dataset
#python3 pyscripts/context_window.py --config $CONFIG

# Run model 
python3 pyscripts/real_data_gibbs.py --config $CONFIG

date
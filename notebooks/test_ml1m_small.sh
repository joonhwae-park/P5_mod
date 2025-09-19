#!/bin/bash
#SBATCH --job-name=ml1m_small       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=l4-1-gm24-c16-m64          # partition name
#SBATCH --gpus=1                       # Enter no.of gpus needed
#SBATCH --output=test_ml1m_small_temp.out          # Name of the output file
#SBATCH --error=test_ml1m_small_temp.err           # Name of the error file
#SBATCH --mem=60G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.ed   # Replace mailid
conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate detic

python test_ml1m_small.py

#!/bin/bash
#SBATCH --job-name=beauty_small       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=l4-8-gm192-c192-m768          # partition name
#SBATCH --gpus=2                       # Enter no.of gpus needed
#SBATCH --output=test_beauty_small.out          # Name of the output file
#SBATCH --error=test_beauty_small.err           # Name of the error file
#SBATCH --mem=128G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.ed   # Replace mailid
conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate detic

python test_beauty_small.py
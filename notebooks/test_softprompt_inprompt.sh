#!/bin/bash
#SBATCH --job-name=mvt_test       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=l4-4-gm96-c48-m192          # partition name
#SBATCH --gpus=1                       # Enter no.of gpus needed
#SBATCH --output=test_softprompt.out          # Name of the output file
#SBATCH --error=test_softprompt.err           # Name of the error file
#SBATCH --mem=48G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.ed   # Replace mailid
conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate detic

python test_softprompt_inprompt.py --csv /scratch/jpa2742/P5/notebooks/ratings_64.csv --out test_softprompt_result.csv

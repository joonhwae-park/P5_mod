#!/bin/bash
#SBATCH --job-name=p5_yelp       # Name of the job
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --partition=a100-8-gm320-c96-m1152        # partition name
#SBATCH --gpus=4                       # Enter no.of gpus needed
#SBATCH --output=p5_yelp.out          # Name of the output file
#SBATCH --error=p5_yelp.err           # Name of the error file
#SBATCH --mem=128G                    # Memory Needed
#SBATCH --mail-type=end                # send mail when job ends
#SBATCH --mail-type=fail               # send mail if job fails
#SBATCH --mail-user=jpa2742@emory.ed   # Replace mailid

conda init bash > /dev/null 2>&1
source /users/jpa2742/.bashrc
conda activate detic

export CUDA_VISIBLE_DEVICES=0,1,2,3

export CUDA_LAUNCH_BLOCKING=1

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1


NPROC=4

name=yelp-small

output=snap/$name
 

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    --master_port 13579 \
    ../src/pretrain.py \
        --distributed --multiGPU \
        --seed 2022 \
        --train yelp \
        --valid yelp \
        --batch_size 32 \
        --optim adamw \
        --warmup_ratio 0.02 \
        --lr 1e-3 \
        --num_workers 4 \
        --clip_grad_norm 1.0 \
        --losses 'rating,sequential,explanation,review,traditional' \
        --backbone 't5-small' \
        --output $output ${@:2} \
        --epoch 10 \
        --max_text_length 512 \
        --gen_max_length 64 \
        --whole_word_embed > $name.log

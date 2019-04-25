#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Tensorboard
pkill tensorboard
rm -rf logs/tb*
tensorboard --logdir logs/ &

# Virtualenv
cd $DIR
virtualenv venv
source venv/bin/activate
pip3.6 install -r requirements.txt

# Train tf 
print_header "Training network"
cd $DIR

# # Comment for using GPU
# export CUDA_VISIBLE_DEVICES=-1

# Experiment
python3.6 main.py \
--env-name 2DNavigation-v0 \
--num-workers 8 \
--fast-lr 0.1 \
--max-kl 0.01 \
--fast-batch-size 20 \
--meta-batch-size 40 \
--num-layers 2 \
--hidden-size 100 \
--gamma 0.99 \
--tau 1.0 \
--cg-damping 1e-5 \
--ls-max-steps 15 \
--device cuda
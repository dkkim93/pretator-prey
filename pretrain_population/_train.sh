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
pip3 install -r requirements.txt

# Add baseline package to path
export PYTHONPATH=$DIR/thirdparty/multiagent-particle-envs:$PYTHONPATH

# Train tf 
print_header "Training network"
cd $DIR

# Comment for using GPU
export CUDA_VISIBLE_DEVICES=-1

# Experiment 1
for i in {0..20..1}
do 
    python3.6 main.py \
    --env-name "simple_tag" \
    --start-timesteps 2000 \
    --ep-max-timesteps 100 \
    --ep-max 1500 \
    --n-predator 2 \
    --n-prey 1 \
    --seed $i \
    --prefix ""
done

# Reinforcement Learning with Model-Agnostic Meta-Learning (MAML)

![HalfCheetahDir](https://raw.githubusercontent.com/tristandeleu/pytorch-maml-rl/master/_assets/halfcheetahdir.gif)

Implementation of Model-Agnostic Meta-Learning (MAML) applied on Reinforcement Learning problems in Pytorch. This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), [Finn et al., 2017](https://arxiv.org/abs/1703.03400)): multi-armed bandits, tabular MDPs, continuous control with MuJoCo, and 2D navigation task.

## Prerequisite
This code uses Python version of 3.6.
Python dependencies will be handled by python virtual environment.  
For non-python dependencies, please install the following:
```
sudo apt-get install libglew-dev patchelf
```

To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```
pip3.6 install --upgrade virtualenv
```

## Usage
You can use the [`_train.sh`](_train.sh) script in order to run reinforcement learning experiments with MAML. This script was tested with Python 3.6. 

## TODO
* Add random seed
* First-order MAML implementation
* Update README.md
* Mujoco visualization time to time

## References
This code is cleaned and modified from the [PyTorch MAML implementation](https://github.com/tristandeleu/pytorch-maml-rl).
This project is, for the most part, a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/). These experiments are based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep
networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

If you want to cite this paper
```
@article{DBLP:journals/corr/FinnAL17,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {Model-{A}gnostic {M}eta-{L}earning for {F}ast {A}daptation of {D}eep {N}etworks},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```

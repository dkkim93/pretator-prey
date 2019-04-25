import torch
import os
import argparse
import numpy as np
import multiprocessing as mp
from maml_rl.metalearner import MetaLearner
from maml_rl.metatester import MetaTester
from maml_rl.policies import NormalMLPPolicy
from maml_rl.baseline import LinearFeatureBaseline
from maml_rl.sampler import BatchSampler
from misc.utils import set_log
from tensorboardX import SummaryWriter


def main(args):
    # Setup for logging
    tb_writer = SummaryWriter('./logs/tb_{}'.format(args.log_name))  # Tensorboard logging
    log = set_log(args)

    # Setup before meta-train starts
    sampler = BatchSampler(
        env_name=args.env_name, 
        batch_size=args.fast_batch_size, 
        num_workers=args.num_workers,
        args=args)

    policy = NormalMLPPolicy(
        input_size=int(np.prod(sampler.envs.observation_space.shape)),
        output_size=int(np.prod(sampler.envs.action_space.shape)),
        hidden_sizes=(args.hidden_size,) * args.num_layers)

    baseline = LinearFeatureBaseline(
        input_size=int(np.prod(sampler.envs.observation_space.shape)))

    meta_learner = MetaLearner(
        sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device,
        args=args, log=log, tb_writer=tb_writer)

    meta_tester = MetaTester(
        sampler, policy, baseline, gamma=args.gamma,
        fast_lr=args.fast_lr, tau=args.tau, device=args.device,
        args=args, log=log, tb_writer=tb_writer)

    # Meta-train starts
    iteration = 0
    while True:
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        episodes = meta_learner.sample(tasks, first_order=args.first_order, iteration=iteration)

        # Train meta-policy
        meta_learner.step(episodes=episodes, args=args)

        # Test meta-policy
        if iteration % 10 == 0:
            test_tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
            meta_tester.few_shot_adaptation(
                meta_policy=meta_learner.policy, tasks=test_tasks, 
                first_order=args.first_order, iteration=iteration)

        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reinforcement learning with Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument(
        '--env-name', type=str,
        help='name of the environment')
    parser.add_argument(
        '--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument(
        '--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument(
        '--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Policy network (relu activation function)
    parser.add_argument(
        '--hidden-size', type=int, default=100,
        help='number of hidden units per layer')
    parser.add_argument(
        '--num-layers', type=int, default=2,
        help='number of hidden layers')

    # Task-specific
    parser.add_argument(
        '--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument(
        '--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument(
        '--meta-batch-size', type=int, default=40,
        help='number of tasks per batch')
    parser.add_argument(
        '--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument(
        '--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument(
        '--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument(
        '--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument(
        '--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Predator
    parser.add_argument(
        "--n-predator", default=1, type=int, 
        help="Number of predators")

    # Prey
    parser.add_argument(
        "--n-prey", default=1, type=int, 
        help="Number of preys")

    # Miscellaneous
    parser.add_argument(
        '--log-name', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument(
        '--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')

    # Device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Set log name
    args.log_name = "env::{}".format(args.env_name)

    main(args)

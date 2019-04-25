import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from trainer.train import train


def set_policy(env, tb_writer, log, args, name, i_agent):
    if name == "predator":
        from policy.predator import Predator
        policy = Predator(env=env, log=log, tb_writer=tb_writer, name=name, args=args, i_agent=i_agent)
    elif name == "prey":
        from policy.prey import Prey
        policy = Prey(env=env, log=log, tb_writer=tb_writer, name=name, args=args, i_agent=i_agent)
    else:
        raise ValueError("Invalid name")

    return policy


def main(args):
    # Create dir
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Create env
    env = make_env(args)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy
    predator_agents = [
        set_policy(env, tb_writer, log, args, name="predator", i_agent=i_agent)
        for i_agent in range(args.n_predator)]

    prey_agents = [
        set_policy(env, tb_writer, log, args, name="prey", i_agent=i_agent)
        for i_agent in range(args.n_prey)]

    # Start training
    train(
        predator_agents=predator_agents, prey_agents=prey_agents, 
        env=env, log=log, tb_writer=tb_writer, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # TD3 Algorithm
    parser.add_argument(
        "--discount", default=0.99, type=float, 
        help="Discount factor")
    parser.add_argument(
        "--tau", default=0.01, type=float, 
        help="Target network update rate")
    parser.add_argument(
        "--start-timesteps", default=1e4, type=int, 
        help="How many time steps purely random policy is run for")
    parser.add_argument(
        "--expl-noise", default=0.1, type=float, 
        help="Std of Gaussian exploration noise")
    parser.add_argument(
        "--batch-size", default=50, type=int, 
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--policy-noise", default=0.2, type=float, 
        help="Noise added to target policy during critic update")
    parser.add_argument(
        "--noise-clip", default=0.5, type=float, 
        help="Range to clip target policy noise")
    parser.add_argument(
        "--policy-freq", default=2, type=int,
        help="Frequency of delayed policy updates")
    parser.add_argument(
        "--actor-lr", default=0.0001, type=float,
        help="Learning rate for actor")
    parser.add_argument(
        "--critic-lr", default=0.001, type=float,
        help="Learning rate for critic")
    parser.add_argument(
        "--grad-clip", default=0.5, type=float,
        help="Gradient clipping to prevent explosion")

    # Predator
    parser.add_argument(
        "--predator-n-hidden", default=400, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--n-predator", default=1, type=int, 
        help="Number of predators")

    # Prey
    parser.add_argument(
        "--prey-n-hidden", default=400, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--n-prey", default=1, type=int, 
        help="Number of preys")

    # Env
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", type=int, required=True,
        help="Episode is terminated when max timestep is reached.")

    # Misc
    parser.add_argument(
        "--ep-max", type=int, required=True,
        help="Training is terminated when max ep is reached")
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_n_predator::%s_n_prey::%s_prefix::%s_log" % (
            args.env_name, str(args.seed), args.n_predator, args.n_prey, args.prefix)

    main(args=args)

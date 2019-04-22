import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################
# MISC
def split_observations(env_observations, args): 
    """
    NOTE that env.observation_space returns observation space for
    both predator and prey but with the order:
    [predator_1, predator_2, ..., prey_1]
    """
    assert len(env_observations) == (args.n_prey + args.n_predator), "Number of obs does not equal"

    predator_observations = env_observations[0:args.n_predator]
    prey_obs = env_observations[args.n_predator:]

    return predator_observations, prey_obs

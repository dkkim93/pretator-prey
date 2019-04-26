import git
import torch
import multiprocessing as mp
import multiagent.scenarios as scenarios
import numpy as np
from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from multiagent.environment import MultiAgentEnv


def make_env(args):
    def check_github(path, branch_name):
        """Checks whether the path has a correct branch name"""
        repo = git.Repo(path)
        branch = repo.active_branch
        assert branch.name == branch_name, "Branch name does not equal the desired branch"

    def _make_env():
        """Load multi-agent particle environment
        This code is modified from: https://github.com/openai/maddpg/blob/master/experiments/train.py
        """
        # Check github branch
        check_github(
            path="./thirdparty/multiagent-particle-envs",
            branch_name="predator_prey")

        # Load multi-agent particle env
        scenario = scenarios.load(args.env_name + ".py").Scenario()
        world = scenario.make_world(
            n_prey=args.n_prey,
            n_predator=args.n_predator)
        done_callback = None

        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            done_callback=done_callback)

        assert env.discrete_action_space is False, "For cont. action, this flag must be False"
        assert env.shared_reward is False, "For predator-prey, this must be False"

        return env
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1, args=None):
        self.env_name = env_name
        self.batch_size = batch_size  # NOTE # of trajectories in each env
        self.num_workers = num_workers
        self.args = args
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv(
            [make_env(args) for _ in range(num_workers)], queue=self.queue)
        self._env = make_env(args)()

    def sample(self, policy, params=None, opponent_policy=None, gamma=0.95, device='cpu'):
        """Sample # of trajectories defined by "self.batch_size". The size of each
        trajectory is defined by the Gym env registration defined at:
        ./maml_rl/envs/__init__.py
        """
        assert opponent_policy is not None

        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, worker_ids = self.envs.reset()  # TODO reset needs to be fixed
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                # Get observations
                predator_observations, prey_observations = self.split_observations(observations)
                predator_observations_torch = torch.from_numpy(predator_observations).to(device=device)
                prey_observations_torch = torch.from_numpy(prey_observations).to(device=device)

                # Get actions
                predator_actions = policy(predator_observations_torch, params=params).sample()
                predator_actions = predator_actions.cpu().numpy()

                prey_actions = opponent_policy.select_deterministic_action(prey_observations_torch)
                prey_actions = prey_actions.cpu().numpy()
            actions = np.concatenate([prey_actions, prey_actions], axis=1)
            new_observations, rewards, dones, new_worker_ids, _ = self.envs.step(actions)
            dones = dones[:, 0]

            # Get new observations
            new_predator_observations, new_prey_observations = self.split_observations(new_observations)

            # Get rewards
            predator_rewards = rewards[:, 0]
            episodes.append(
                predator_observations, 
                predator_actions, 
                predator_rewards,
                worker_ids)
            observations, worker_ids = new_observations, new_worker_ids

        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        n_population = 1  # TODO Change number later
        i_agents = np.random.randint(low=0, high=n_population, size=(num_tasks, ))
        tasks = [{"i_agent": i_agent} for i_agent in i_agents]
        return tasks

    def split_observations(self, observations):
        predator_observations = []
        prey_observations = []
        for obs in observations:
            assert len(obs) == 2
            predator_observations.append(obs[0])
            prey_observations.append(obs[1])

        return \
            np.asarray(predator_observations, dtype=np.float32), \
            np.asarray(prey_observations, dtype=np.float32)

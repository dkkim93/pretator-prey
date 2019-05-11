import copy
import git
import torch
import multiprocessing as mp
import multiagent.scenarios as scenarios
import numpy as np
from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from multiagent.environment import MultiAgentEnv


def make_env(args, i_worker):
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
        done_callback = scenario.done_callback

        env = MultiAgentEnv(
            world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            done_callback=done_callback)

        print("i_worker:", i_worker)
        env.seed(i_worker)

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
            [make_env(args, i_worker) for i_worker in range(num_workers)], queue=self.queue)
        self._env = make_env(args, i_worker=99)()

    def sample(self, policy, params=None, predators=None, gamma=0.95, device='cpu'):
        """Sample # of trajectories defined by "self.batch_size". The size of each
        trajectory is defined by the Gym env registration defined at:
        ./maml_rl/envs/__init__.py
        """
        assert predators is not None

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
                predator0_observations, predator1_observations, prey_observations = \
                    self.split_observations(observations)
                predator0_observations_torch = torch.from_numpy(predator0_observations).to(device=device)
                predator1_observations_torch = torch.from_numpy(predator1_observations).to(device=device)
                prey_observations_torch = torch.from_numpy(prey_observations).to(device=device)

                # Get actions
                predator0_actions = predators[0].select_deterministic_action(predator0_observations_torch)
                predator0_actions = predator0_actions.cpu().numpy()

                predator1_actions = predators[1].select_deterministic_action(predator1_observations_torch)
                predator1_actions = predator1_actions.cpu().numpy()

                prey_actions = policy(prey_observations_torch, params=params).sample()
                prey_actions = prey_actions.cpu().numpy()

            actions = np.concatenate([predator0_actions, predator1_actions, prey_actions], axis=1)
            new_observations, rewards, dones, new_worker_ids, _ = self.envs.step(copy.deepcopy(actions))
            dones = dones[:, -1]

            # Get new observations
            # _, _, new_prey_observations = self.split_observations(new_observations)

            # Get rewards
            prey_rewards = rewards[:, -1]
            episodes.append(
                prey_observations, 
                prey_actions, 
                prey_rewards,
                worker_ids)
            observations, worker_ids = new_observations, new_worker_ids

        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks, test=False):
        if test is False:
            i_agents = np.random.randint(low=0, high=16, size=(num_tasks, ))
        else:
            i_agents = np.random.randint(low=16, high=21, size=(num_tasks, ))

        tasks = [{"i_agent": i_agent} for i_agent in i_agents]
        return tasks

    def split_observations(self, observations):
        predator0_observations = []
        predator1_observations = []
        prey_observations = []

        for obs in observations:
            assert len(obs) == 3
            predator0_observations.append(obs[0])
            predator1_observations.append(obs[1])
            prey_observations.append(obs[2])

        return \
            np.asarray(predator0_observations, dtype=np.float32), \
            np.asarray(predator1_observations, dtype=np.float32), \
            np.asarray(prey_observations, dtype=np.float32)

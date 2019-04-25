import git
import torch
import multiprocessing as mp
import multiagent.scenarios as scenarios
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

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        """Sample # of trajectories defined by "self.batch_size". The size of each
        trajectory is defined by the Gym env registration defined at:
        ./maml_rl/envs/__init__.py
        """
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, worker_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_worker_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, worker_ids)
            observations, worker_ids = new_observations, new_worker_ids

        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks

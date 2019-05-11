import copy
import torch
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence
from maml_rl.utils.torch_utils import weighted_mean, detach_distribution, weighted_normalize
from misc.utils import total_rewards


class MetaTester(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes 
    (before and after the one-step adaptation), compute the inner loss, compute 
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, 
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized 
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, 
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """
    def __init__(self, sampler, policy, baseline, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu', 
                 args=None, log=None, tb_writer=None):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)
        self.args = args
        self.log = log
        self.tb_writer = tb_writer

        assert self.log is not None, "args is None"
        assert self.log is not None, "logging is None"
        assert self.tb_writer is not None, "tb_writer is None"

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner 
        loss is REINFORCE with baseline [2], computed on advantages estimated 
        with Generalized Advantage Estimation (GAE, [3]).
        The baseline is subtracted from the empirical return to reduce
        variance of the optimization. In here, a linear function as the 
        baseline with a time-varying feature vector is used.
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0, weights=episodes.mask)

        return loss

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from 
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)

        # Get the loss on the training episodes
        loss = self.inner_loss(episodes)

        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(
            loss, step_size=self.fast_lr,
            first_order=first_order)

        return params

    def few_shot_adaptation(self, meta_policy, tasks, first_order, iteration, predators):
        assert iteration is not None, "iteration is None. Provide value"

        total_shot = 2
        episodes = [[] for _ in range(total_shot)]

        for task in tasks:
            # Copy policy from meta-train policy
            self.policy = copy.deepcopy(meta_policy)

            # Each task is defined as a different opponent
            predators[0].load_model(
                filename="seed::" + str(task["i_agent"]) + "_predator0",
                directory="./pytorch_models/1vs2/")

            predators[1].load_model(
                filename="seed::" + str(task["i_agent"]) + "_predator1",
                directory="./pytorch_models/1vs2/")

            for k_shot in range(total_shot):
                if k_shot == 0:
                    params = None

                # Get task-specific train data (line 5)
                # train_episodes shape: (horizon, args.fast_batch_size)
                train_episodes = self.sampler.sample(
                    policy=self.policy, params=params, predators=predators,
                    gamma=self.gamma, device=self.device)

                # Compute task-specific adapted parameters (line 6-7)
                params = self.adapt(train_episodes, first_order=first_order)

                # Continue updating policy for K-shot adaptation
                self.policy.load_state_dict(params)

                episodes[k_shot].append((train_episodes))

        # Log performance
        for k_shot in range(total_shot):
            reward = total_rewards([ep.rewards for ep in episodes[k_shot]])

            self.log[self.args.log_name].info(
                "[ TEST ] Episode {0}: Reward (K-shot {1}): {2}".format(iteration, k_shot, reward))
            self.tb_writer.add_scalars(
                main_tag='test/total_rewards', 
                tag_scalar_dict={"k_shot_" + str(k_shot): reward}, 
                global_step=iteration)

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""
        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(
                kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, pis = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)
            with torch.set_grad_enabled(old_pi is None):
                pi = self.policy(valid_episodes.observations, params=params)
                pis.append(detach_distribution(pi))

                if old_pi is None:
                    old_pi = detach_distribution(pi)

                values = self.baseline(valid_episodes)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages, weights=valid_episodes.mask)

                log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), pis)

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device

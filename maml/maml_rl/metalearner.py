import torch
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.distributions.kl import kl_divergence
from maml_rl.utils.torch_utils import weighted_mean, detach_distribution, weighted_normalize
from maml_rl.utils.optimization import conjugate_gradient
from misc.utils import total_rewards


class MetaLearner(object):
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

    def sample(self, tasks, prey, first_order=False, iteration=None):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        assert iteration is not None, "iteration is None. Provide value"

        episodes = []
        for task in tasks:
            # Each task is defined as a different opponent
            prey.load_model(
                filename="seed::" + str(task["i_agent"]) + "_prey0",
                directory="./pytorch_models/1vs1/")

            # Get task-specific train data (line 5)
            train_episodes = self.sampler.sample(
                policy=self.policy, params=None, prey=prey,
                gamma=self.gamma, device=self.device)
            
            # Compute task-specific adapted parameters (line 6-7)
            params = self.adapt(train_episodes, first_order=first_order)

            # Get meta data for meta-policy training (line 8)
            valid_episodes = self.sampler.sample(
                self.policy, params=params, prey=prey,
                gamma=self.gamma, device=self.device)

            episodes.append((train_episodes, valid_episodes))

        # Log performance
        reward_before_update = total_rewards([ep.rewards for ep, _ in episodes])
        reward_after_update = total_rewards([ep.rewards for _, ep in episodes])

        self.log[self.args.log_name].info("Episode {0}: Reward before: {1}".format(iteration, reward_before_update))
        self.log[self.args.log_name].info("Episode {0}: Reward after: {1}".format(iteration, reward_after_update))
        self.tb_writer.add_scalars(
            main_tag='train/total_rewards', 
            tag_scalar_dict={
                "before_update": reward_before_update, 
                "after_update": reward_after_update}, 
            global_step=iteration)

        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        """In Trust Region Policy Optimization (TRPO, [4]), the heuristic
        approximation which considers the "average" KL divergence is used instead
        """
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
        """Hessian-vector product, based on the Perlmutter method"""
        def _product(vector):
            kl = self.kl_divergence(episodes, old_pis=None)
            grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector
        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        """Computes the surrogate loss in TRPO:
        (pi(a|s) / q(a|s)) * Q(s,a) in Eqn 14
        Because the meta-loss tried to find theta that minimizes
        loss with phi, the loss is computed with valid episodes  
        """
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
                ratio = torch.exp(log_ratio)  # Convert back to ratio from log

                loss = -weighted_mean(ratio * advantages, dim=0, weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
                kls.append(kl)

        return (
            torch.mean(torch.stack(losses, dim=0)),
            torch.mean(torch.stack(kls, dim=0)), 
            pis)

    def step(self, episodes, args):
        """Meta-optimization step (ie. update of the initial parameters), based 
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        # Compute initial surrogate loss assuming old_pi and pi are the same
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)
        
        if args.first_order:
            raise ValueError("no first order")
            step = grads
        else:
            # Compute the step direction with Conjugate Gradient
            hessian_vector_product = self.hessian_vector_product(episodes, damping=args.cg_damping)
            stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=args.cg_iters)

            # Compute the Lagrange multiplier
            shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
            lagrange_multiplier = torch.sqrt(shs / args.max_kl)

            step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(args.ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < args.max_kl):
                break
            step_size *= args.ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.device = device

    def save(self, iteration):
        filename = "theta_" + str(iteration)
        directory = "./pytorch_models"
        torch.save(self.policy.state_dict(), '%s/%s_actor.pth' % (directory, filename))

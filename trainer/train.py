import copy
import numpy as np
from misc.train_utils import *

total_timesteps = 0
total_eps = 0


def eval_progress(predator_agents, prey_agents, env, log, tb_writer, args):
    if total_eps % 3 == 0:
        predator_reward = 0.
        prey_reward = 0.
        n_eval = 5

        for i_eval in range(n_eval):
            env_observations = env.reset()
            ep_timesteps = 0.
            terminal = False

            while True:
                if total_eps % 20 == 0:
                    env.render()

                predator_observations, prey_observations = \
                    split_observations(env_observations, args=args)

                predator_actions = []
                for predator, predator_obs in zip(predator_agents, predator_observations):
                    predator_action = predator.select_deterministic_action(np.array(predator_obs))
                    predator_actions.append(predator_action)

                prey_actions = []
                for prey, prey_obs in zip(prey_agents, prey_observations):
                    prey_action = prey.select_deterministic_action(np.array(prey_obs))
                    prey_actions.append(prey_action)

                new_env_obseravations, env_rewards, dones, _ = env.step(copy.deepcopy(predator_actions + prey_actions))
                terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False

                # For next timestep
                env_observations = new_env_obseravations
                predator_reward += env_rewards[0]
                prey_reward += env_rewards[-1]
                ep_timesteps += 1

                if terminal:
                    break

        # Log result
        predator_reward /= float(n_eval)
        prey_reward /= float(n_eval)

        log[args.log_name].info(
            "[ EVAL ] Predator Reward {:.5f} at episode {}".format(predator_reward, total_eps))
        log[args.log_name].info(
            "[ EVAL ] Prey Reward {:.5f} at episode {}".format(prey_reward, total_eps))
        tb_writer.add_scalars(
            "reward/predator_reward", 
            {"eval_reward": predator_reward}, total_eps)
        tb_writer.add_scalars(
            "reward/prey_reward", 
            {"eval_reward": prey_reward}, total_eps)


def collect_one_traj(predator_agents, prey_agents, env, log, args, tb_writer):
    global total_timesteps, total_eps

    ep_predator_reward = 0.
    ep_prey_reward = 0.
    ep_timesteps = 0
    env_observations = env.reset()

    while True:
        predator_observations, prey_observations = \
            split_observations(env_observations, args=args)

        # Predator selects its action
        predator_actions = []
        for predator, predator_obs in zip(predator_agents, predator_observations):
            predator_action = predator.select_stochastic_action(np.array(predator_obs), total_timesteps)
            predator_actions.append(predator_action)

        # Prey selects its action
        prey_actions = []
        for prey, prey_obs in zip(prey_agents, prey_observations):
            prey_action = prey.select_stochastic_action(np.array(prey_obs), total_timesteps)
            prey_actions.append(prey_action)

        # Perform action
        new_env_observations, env_rewards, dones, _ = env.step(copy.deepcopy(predator_actions + prey_actions))
        terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False

        # Add predator memory
        # NOTE Predator uses centralized training (MADDPG)
        new_predator_observations, new_prey_observations = \
            split_observations(new_env_observations, args=args)
        predator_reward = env_rewards[0]

        for i_predator, predator in enumerate(predator_agents):
            predator.add_memory(
                obs=predator_observations + prey_observations,
                new_obs=new_predator_observations + prey_observations,
                action=predator_actions + prey_actions,
                reward=[predator_reward for _ in range(len(dones))],
                done=[False for _ in range(len(dones))])

        # Add prey memory
        # NOTE Predator uses decentralized training (DDPG)
        prey_reward = env_rewards[-1]

        for i_prey, prey in enumerate(prey_agents):
            prey.add_memory(
                obs=prey_observations[i_prey],
                new_obs=new_prey_observations[i_prey],
                action=prey_actions[i_prey],
                reward=prey_reward,
                done=False)

        # For next timestep
        env_observations = new_env_observations
        ep_timesteps += 1
        total_timesteps += 1
        ep_predator_reward += predator_reward
        ep_prey_reward += prey_reward

        if terminal: 
            total_eps += 1

            log[args.log_name].info("Train episode predator reward {} at episode {}".format(
                ep_predator_reward, total_eps))
            log[args.log_name].info("Train episode prey reward {} at episode {}".format(
                ep_prey_reward, total_eps))
            tb_writer.add_scalars(
                "reward/predator_reward", 
                {"train_reward": ep_predator_reward}, total_eps)
            tb_writer.add_scalars(
                "reward/prey_reward", 
                {"train_reward": ep_prey_reward}, total_eps)

            return


def train(predator_agents, prey_agents, env, log, tb_writer, args):
    while True:
        eval_progress(
            predator_agents=predator_agents, prey_agents=prey_agents,  
            env=env, log=log, tb_writer=tb_writer, args=args)
        
        collect_one_traj(
            predator_agents=predator_agents, prey_agents=prey_agents,
            env=env, log=log, args=args, tb_writer=tb_writer)

        for predator in predator_agents:
            predator.update_policy(
                agents=predator_agents + prey_agents, 
                total_eps=total_eps)

        for prey in prey_agents:
            prey.update_policy(total_eps=total_eps)

        # if args.save_opponent and total_eps % 500 == 0:
        #     save(opponent_n, total_eps)

import copy
import numpy as np
from misc.train_utils import *

total_timesteps = 0
total_eps = 0


def eval_progress(predator_agents, prey_agents, env, log, tb_writer, args):
    if total_eps % 3 == 0:
        predator_reward = 0.
        prey_reward = 0.
        n_eval = 10

        for i_eval in range(n_eval):
            env_observations = env.reset()
            ep_timesteps = 0.
            terminal = False

            while True:
                # if total_eps % 100 == 0:
                #     env.render()

                predator_observations, prey_observations = \
                    split_observations(env_observations, args=args)

                predator_actions = []
                for predator, predator_obs in zip(predator_agents, predator_observations):
                    predator_action = predator.select_deterministic_action(np.array(predator_obs))
                    predator_action = np.array([0., 0.])
                    predator_actions.append(predator_action)

                prey_actions = []
                for prey, prey_obs in zip(prey_agents, prey_observations):
                    prey_action = prey.select_deterministic_action(np.array(prey_obs))
                    prey_actions.append(prey_action)

                new_env_obseravations, rewards, dones, _ = env.step(copy.deepcopy(predator_actions + prey_actions))
                terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False

                # For next timestep
                env_observations = new_env_obseravations
                predator_reward += rewards[0]
                prey_reward += rewards[-1]
                ep_timesteps += 1

                if terminal:
                    break

        # Log result
        predator_reward /= float(n_eval)
        prey_reward /= float(n_eval)

        log[args.log_name].info("[ EVAL ] Predator Reward {:.5f} at episode {}".format(predator_reward, total_eps))
        log[args.log_name].info("[ EVAL ] Prey Reward {:.5f} at episode {}".format(prey_reward, total_eps))
        tb_writer.add_scalar("reward/predator_reward", predator_reward, total_eps)
        tb_writer.add_scalar("reward/prey_reward", prey_reward, total_eps)


def collect_one_traj(opponent_n, env, log, args, tb_writer):
    global total_timesteps, total_eps

    ep_reward = 0.
    ep_timesteps = 0
    env_obs_n = env.reset()

    while True:
        # opponent selects its action
        opponent_obs_n = env_obs_n
        opponent_action_n = []
        for opponent, opponent_obs in zip(opponent_n, opponent_obs_n):
            opponent_action = opponent.select_stochastic_action(
                obs=np.array(opponent_obs), total_timesteps=total_timesteps)
            opponent_action_n.append(opponent_action)

        # Perform action
        new_env_obs_n, env_reward_n, env_done_n, _ = env.step(copy.deepcopy(opponent_action_n))
        terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False

        # Add opponent memory
        new_opponent_obs_n = new_env_obs_n
        opponent_reward_n = env_reward_n

        for i_opponent, opponent in enumerate(opponent_n):
            opponent.add_memory(
                obs=opponent_obs_n[i_opponent],
                new_obs=new_opponent_obs_n[i_opponent],
                action=opponent_action_n[i_opponent],
                reward=opponent_reward_n[i_opponent],
                done=False)

        # For next timestep
        env_obs_n = new_env_obs_n
        ep_timesteps += 1
        total_timesteps += 1
        ep_reward += env_reward_n[0]

        if terminal: 
            total_eps += 1
            log[args.log_name].info("Train episode reward {} at episode {}".format(ep_reward, total_eps))
            tb_writer.add_scalar("reward/train_ep_reward", ep_reward, total_eps)

            return ep_reward


def train(predator_agents, prey_agents, env, log, tb_writer, args):
    while True:
        eval_progress(
            predator_agents=predator_agents, prey_agents=prey_agents,  
            env=env, log=log, tb_writer=tb_writer, args=args)
        
        import sys
        sys.exit()

        collect_one_traj(
            opponent_n=opponent_n, env=env, log=log,
            args=args, tb_writer=tb_writer)

        for opponent in opponent_n:
            opponent.update_policy(opponent_n, total_timesteps)

        if args.save_opponent and total_eps % 500 == 0:
            save(opponent_n, total_eps)

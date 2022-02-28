import gym
import mujoco_gym
import numpy as np
import csv
import importlib
import multiprocessing, ray
from matplotlib import pyplot as plt
from mujoco_gym.learn import load_policy
import argparse

ENV_NAME = "Reacher-v2"

# NOTE: Most of this is shamelessly copied from render_policy in learn.py.
# Link: https://github.com/Healthcare-Robotics/assistive-gym/blob/fb799c377e1f144ff96044fb9096725f7f9cfc61/assistive_gym/learn.py#L96


def make_env(env_name, seed=1001):
    env = gym.make(env_name)
    env.seed(seed)
    return env


def generate_rollout_data(data_dir, seed, num_rollouts, augmented, render):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)

    # Set up the environment
    env = make_env(ENV_NAME, seed=seed)  # fixed seed for reproducibility (1000 for training, 1001 for testing)

    # Load pretrained policy from file
    algo = 'sac'

    # 13390 iterations total for expert RL -->
    # 0.01: 130
    # 0.05: 670
    # 0.1: 1340
    # 0.2: 2680
    # 0.8: 10710
    # 1.0: 13390

    checkpoints = ["/home/jeremy/gym/trained_models/sac4096/sac/Reacher-v2/checkpoint_000130/checkpoint-130",
                   "/home/jeremy/gym/trained_models/sac4096/sac/Reacher-v2/checkpoint_000670/checkpoint-670",
                   "/home/jeremy/gym/trained_models/sac4096/sac/Reacher-v2/checkpoint_001340/checkpoint-1340",
                   "/home/jeremy/gym/trained_models/sac4096/sac/Reacher-v2/checkpoint_002680/checkpoint-2680",
                   "/home/jeremy/gym/trained_models/sac4096/sac/Reacher-v2/checkpoint_010710/checkpoint-10710",
                   "/home/jeremy/gym/trained_models/sac4096/sac/Reacher-v2/checkpoint_013390/checkpoint-13390"
                   ]

    if render:
        env.render()

    demos = []  # collection of trajectories
    total_rewards = []  # final reward at the end of a trajectory/demo
    rewards_over_time = []  # rewards at each timestep for each trajectory
    cum_rewards_over_time = []  # cumulative reward at each timestep for each trajectory, with a separate checkpoint dimension
    rewards_per_checkpoint_level = []  # final reward at the end of each trajectory, with a separate checkpoint dimension
    for i, checkpoint_path in enumerate(checkpoints):
        checkpoint_agent, _ = load_policy(env, algo, ENV_NAME, checkpoint_path, seed=seed)

        cum_rewards_over_time.append([])
        rewards_per_checkpoint_level.append([])

        num_demos = num_rollouts
        for demo in range(num_demos):
            traj = []
            cum_reward_over_time = []
            reward_over_time = []
            total_reward = 0
            observation = env.reset()
            info = None
            done = False
            while not done:
                # This env.render() is necessary to render each frame in MuJoCo
                if render:
                    env.render()

                # Compute the next action using the trained policy
                action = checkpoint_agent.compute_action(observation)

                # Collect the data
                # print("Observation:", observation)
                # print("Action:", action)

                # Reacher privileged features: end effector - target distance
                if ENV_NAME == "Reacher-v2":
                    distance = np.linalg.norm(observation[8:11])
                    handpicked_features = np.array([distance])

                if augmented:
                    data = np.concatenate((observation, handpicked_features))
                else:
                    data = observation

                # Step the simulation forward using the action from our trained policy
                observation, reward, done, info = env.step(action)

                traj.append(data)
                total_reward += reward
                reward_over_time.append(reward)
                cum_reward_over_time.append(total_reward)
                # print("Reward:", reward)
                # print("Task Success:", info['task_success'])
                # print("\n")
            demos.append(traj)

            cum_rewards_over_time[i].append(cum_reward_over_time)
            rewards_per_checkpoint_level[i].append(total_reward)

            # print(total_reward)
            total_rewards.append(total_reward)
            rewards_over_time.append(reward_over_time)

    rewards_per_checkpoint_level = np.array(rewards_per_checkpoint_level)
    mean_rewards_per_checkpoint_level = np.mean(rewards_per_checkpoint_level, axis=1)

    demos = np.asarray(demos)
    total_rewards = np.asarray(total_rewards)
    rewards_over_time = np.asarray(rewards_over_time)
    # print(demos)
    # print(total_rewards)

    np.save(data_dir+"/demos.npy", demos)
    np.save(data_dir+"/demo_rewards.npy", total_rewards)
    np.save(data_dir+"/demo_reward_per_timestep.npy", rewards_over_time)


    with np.printoptions(precision=3):
        print(rewards_per_checkpoint_level)
        print(mean_rewards_per_checkpoint_level)

    # Code for plotting how cumulative reward changes over time with each level of checkpoint.
    # plt.figure()
    #
    # plt.subplot(231)
    # for p in cum_rewards_over_time[0]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(232)
    # for p in cum_rewards_over_time[1]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(233)
    # for p in cum_rewards_over_time[2]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(234)
    # for p in cum_rewards_over_time[3]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(235)
    # for p in cum_rewards_over_time[4]:
    #     plt.plot(p, 'b')
    #
    # plt.subplot(236)
    # for p in cum_rewards_over_time[5]:
    #     plt.plot(p, 'b')
    #
    # plt.savefig("cum_rewards_over_time.png")
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data_dir', default='', help="location for generated rollouts")
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    parser.add_argument('--num_rollouts', default=20, type=int, help="number of rollouts")
    parser.add_argument('--augmented', dest='augmented', default=False, action='store_true', help="whether data consists of states + linear features pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--render', dest='render', default=False, action='store_true', help="whether to render rollouts")  # NOTE: type=bool doesn't work, value is still true.
    args = parser.parse_args()

    data_dir = args.data_dir
    seed = args.seed
    num_rollouts = args.num_rollouts
    augmented = args.augmented
    render = args.render

    generate_rollout_data(data_dir, seed, num_rollouts, augmented, render)

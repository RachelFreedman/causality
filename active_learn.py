import trex.model
import mujoco_gym.learn
import argparse
import numpy as np
import multiprocessing, ray


def get_rollouts(num_rollouts, policy_path, seed):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    # Set up the environment
    env = mujoco_gym.learn.make_env("Reacher-v2", seed=seed)
    # Load pretrained policy from file
    test_agent, _ = mujoco_gym.learn.load_policy(env, 'sac', "Reacher-v2", policy_path, seed=seed)

    new_rollouts = []
    new_rollout_rewards = []
    for r in range(num_rollouts):
        traj = []
        reward_total = 0.0
        obs = env.reset()
        done = False
        while not done:
            action = test_agent.compute_action(obs)

            distance = np.linalg.norm(obs[8:11])
            action_norm = np.linalg.norm(action)
            privileged_features = np.array([distance, action_norm])

            data = np.concatenate((obs, privileged_features))

            obs, reward, done, info = env.step(action)

            traj.append(data)
            reward_total += reward

        new_rollouts.append(traj)
        new_rollout_rewards.append(reward_total)

    new_rollouts = np.asarray(new_rollouts)
    new_rollout_rewards = np.asarray(new_rollout_rewards)
    return new_rollouts, new_rollout_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help="seed")
    parser.add_argument('--num_al_iter', default=0, type=int, help="number of active learning iterations")
    parser.add_argument('--mix', default=0.5, type=float, help="hyperparameter for how much to mix in new rollouts, where 1 means the next iteration consists of ONLY new rollouts")

    args = parser.parse_args()

    seed = args.seed
    num_al_iter = args.num_al_iter
    mixing_factor = args.mix

    # Load demonstrations from file and initialize pool of demonstrations
    demos = np.load("trex/data/augmented_full/demos.npy")
    demo_rewards = np.load("trex/data/augmented_full/demo_rewards.npy")
    num_demos = demos.size[0]

    # For num_al_iter active learning iterations:
    for i in range(num_al_iter):
        # 1. Run reward learning
        config = "active_learning/augmentedfull_linear_2000prefs_60pairdelta_100epochs_10patience_001lr_001l1reg"
        reward_model_path = "/home/jeremy/gym/trex/models/"+config+"_seed"+seed+".params"
        # Use the al_data argument to input our pool of changing demonstrations
        trex.model.run("/home/jeremy/gym/trex/models/"+config+"_seed"+seed+".params", seed=seed, num_comps=2000, pair_delta=60,
                       num_epochs=100, patience=10, lr=0.01, l1_reg=0.01, al_data=(demos, demo_rewards))

        # 2. Run RL (using the learned reward)
        policy_save_dir = "./trained_models_reward_learning/"+config+"_seed"+seed
        checkpoint_path = mujoco_gym.learn.train("ReacherLearnedReward-v0", "sac", timesteps_total=1000000, save_dir=policy_save_dir, load_policy_path=policy_save_dir, seed=seed, reward_net_path=reward_model_path)

        # 3. Load RL policy, generate rollouts (number depends on mixing factor), and rank according to GT reward
        num_new_rollouts = num_demos * mixing_factor
        new_rollouts, new_rollout_rewards = get_rollouts(num_new_rollouts, checkpoint_path, seed)

        # 4. Based on mixing factor, sample (without replacement) demonstrations from previous iteration accordingly
        num_old_trajs = num_demos * (1 - mixing_factor)
        old_traj_i = np.random.choice(num_demos, size=num_old_trajs, replace=False)
        old_trajs = demos[old_traj_i]
        old_traj_rewards = demo_rewards[old_traj_i]

        # Update our pool of demonstrations
        demos = np.concatenate((old_trajs, new_rollouts), axis=0)
        demo_rewards = np.concatenate((old_traj_rewards, new_rollout_rewards), axis=0)




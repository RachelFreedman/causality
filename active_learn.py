import trex.model
import mujoco_gym.learn
import argparse
import numpy as np
import multiprocessing, ray
import re, string
import sys

# NOTE: Before the script finishes, the user needs to create directories for the policy_eval_dir. 

EVAL_SEED = 3


def get_rollouts(num_rollouts, policy_path, seed, state_action=False, augmented_full=False, augmented=False):
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

            if augmented_full:  # TODO: outdated
                data = np.concatenate((obs, privileged_features))
            elif augmented:  # TODO: outdated
                data = np.concatenate((obs, [privileged_features[0]]))
            elif state_action:
                data = np.concatenate((obs, action))
            else:
                data = obs

            obs, reward, done, info = env.step(action)

            traj.append(data)
            reward_total += reward

        new_rollouts.append(traj)
        new_rollout_rewards.append(reward_total)

    new_rollouts = np.asarray(new_rollouts)
    new_rollout_rewards = np.asarray(new_rollout_rewards)
    return new_rollouts, new_rollout_rewards


def run_active_learning(num_al_iter, mixing_factor, union_rollouts, retrain, seed, nn, reward_epochs_per_iter, rl_steps_per_iter):
    np.random.seed(seed)

    # Load demonstrations from file and initialize pool of demonstrations
    if nn:
        demos = np.load("trex/data/reacher/raw_stateaction/raw_360/demos.npy")
        demo_rewards = np.load("trex/data/reacher/raw_stateaction/raw_360/demo_rewards.npy")
    else:  # TODO: outdated
        demos = np.load("trex/data/augmented_full/demos.npy")
        demo_rewards = np.load("trex/data/augmented_full/demo_rewards.npy")
    num_demos = demos.shape[0]

    if mixing_factor is not None:
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        config = "active_learning/" + str(num_al_iter) + "aliter_" + regex.sub('', str(mixing_factor)) + "mix_"
    elif union_rollouts is not None:
        config = "active_learning/" + str(num_al_iter) + "aliter_" + str(union_rollouts) + "union_"
    if retrain:
        if nn:
            config = config + "retrain_stateaction_hdim128-64_324demos_allpairs_100epochs_10patience_001lr_00001weightdecay_seed" + str(
                seed)
        else:  # TODO: outdated
            config = config + "retrain_augmentedfull_linear_2000prefs_2deltareward_100epochs_10patience_001lr_001l1reg_seed" + str(
                seed)
    else:
        if nn:
            config = config + "stateaction_hdim128-64_324demos_allpairs_100epochs_10patience_001lr_00001weightdecay_seed" + str(
                seed)
        else:  # TODO: outdated
            config = config + "augmentedfull_linear_2000prefs_2deltareward_100epochs_10patience_001lr_001l1reg_seed" + str(
                seed)

    reward_model_path = "/home/jeremy/gym/trex/models/reacher/" + config + ".params"
    reward_output_path = "/home/jeremy/gym/trex/reward_learning_outputs/reacher/" + config + ".txt"

    policy_save_dir = "./trained_models_reward_learning/reacher/" + config
    policy_eval_dir = "/home/jeremy/gym/trex/rl/eval/reacher/" + config

    gt_rewards = []
    learned_rewards = []
    successes = []
    weights = []
    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []
    test_accs = []
    test_losses = []
    # For num_al_iter active learning iterations:
    for i in range(num_al_iter):
        # 1. Run reward learning
        with open(reward_output_path, 'a') as sys.stdout:
            # Use the al_data argument to input our pool of changing demonstrations
            if nn:
                final_weights, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = trex.model.run(
                    "Reacher-v2", reward_model_path, seed=seed, hidden_dims=(128, 64), num_demos=324, all_pairs=True,
                    num_epochs=reward_epochs_per_iter, patience=10, lr=0.01, weight_decay=0.0001, state_action=True,
                    al_data=(demos, demo_rewards), test=True, load_weights=(not retrain), return_weights=False)
            else:  # TODO: outdated ish
                final_weights, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = trex.model.run(
                    "Reacher-v2", reward_model_path, seed=seed, num_comps=2000, delta_reward=2,
                    num_epochs=reward_epochs_per_iter, patience=10, lr=0.01, l1_reg=0.01, augmented_full=True,
                    al_data=(demos, demo_rewards), test=True, load_weights=(not retrain), return_weights=True)
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            test_accs.append(test_acc)
            test_losses.append(test_loss)

        sys.stdout = sys.__stdout__  # reset stdout
        if not nn:  # TODO: outdated ish
            weights.append(final_weights['fcs.0.weight'].cpu().detach().numpy())

        # 2. Run RL (using the learned reward)
        if retrain:
            checkpoint_path = mujoco_gym.learn.train("ReacherLearnedReward-v0", "sac",
                                                     timesteps_total=rl_steps_per_iter, save_dir=policy_save_dir + "/" + str(i+1),
                                                     load_policy_path='', seed=seed,
                                                     reward_net_path=reward_model_path)
        else:
            checkpoint_path = mujoco_gym.learn.train("ReacherLearnedReward-v0", "sac",
                                                     timesteps_total=((i+1)*rl_steps_per_iter), save_dir=policy_save_dir,
                                                     load_policy_path=policy_save_dir, seed=seed,
                                                     reward_net_path=reward_model_path)

        # 3. Load RL policy, generate rollouts (number depends on mixing factor), and rank according to GT reward
        if mixing_factor is not None:
            print("using mixing factor of", mixing_factor, "...")
            num_new_rollouts = round(num_demos * mixing_factor)
        elif union_rollouts is not None:
            print("unioning", union_rollouts, "rollouts...")
            num_new_rollouts = union_rollouts

        if nn:
            new_rollouts, new_rollout_rewards = get_rollouts(num_new_rollouts, checkpoint_path, seed, state_action=True)
        else:
            new_rollouts, new_rollout_rewards = get_rollouts(num_new_rollouts, checkpoint_path, seed, augmented_full=True)

        # 4. Based on mixing factor, sample (without replacement) demonstrations from previous iteration accordingly
        if mixing_factor is not None:
            num_old_trajs = round(num_demos * (1 - mixing_factor))
            old_traj_i = np.random.choice(num_demos, size=num_old_trajs, replace=False)
            old_trajs = demos[old_traj_i]
            old_traj_rewards = demo_rewards[old_traj_i]
        elif union_rollouts is not None:
            old_trajs = demos
            old_traj_rewards = demo_rewards

        # Update our pool of demonstrations
        demos = np.concatenate((old_trajs, new_rollouts), axis=0)
        demo_rewards = np.concatenate((old_traj_rewards, new_rollout_rewards), axis=0)

        # checkpoint_path = policy_save_dir + "/sac/ReacherLearnedReward-v0/checkpoint_0000" + f'{(i+1):02d}' + "/checkpoint-" + str(i+1)
        # 5. Evaluate (latest) trained policy
        eval_path = policy_eval_dir + "/" + str(i+1) + ".txt"
        with open(eval_path, 'w') as sys.stdout:
            gt_mean_reward, gt_std_reward, mean_success, std_success = mujoco_gym.learn.evaluate_policy("Reacher-v2", "sac", checkpoint_path, n_episodes=100, seed=EVAL_SEED, verbose=True)
            learned_mean_reward, learned_std_reward, _, _ = mujoco_gym.learn.evaluate_policy("ReacherLearnedReward-v0", "sac", checkpoint_path, n_episodes=100, seed=EVAL_SEED, verbose=True, reward_net_path=reward_model_path)

        sys.stdout = sys.__stdout__  # reset stdout
        gt_rewards.append([gt_mean_reward, gt_std_reward])
        learned_rewards.append([learned_mean_reward, learned_std_reward])
        successes.append([mean_success, std_success])

    # NOTE: rewards[i] denotes the ith iteration of active learning. rewards[i][0] gives the reward mean,
    # and rewards[i][1] the std dev.
    # weights[i] contains the (linear) reward function weights at the end of the ith iteration.
    train_accs = np.asarray(train_accs)
    train_losses = np.asarray(train_losses)
    val_accs = np.asarray(val_accs)
    val_losses = np.asarray(val_losses)
    test_accs = np.asarray(test_accs)
    test_losses = np.asarray(test_losses)
    gt_rewards = np.asarray(gt_rewards)
    learned_rewards = np.asarray(learned_rewards)
    successes = np.asarray(successes)

    np.save(policy_eval_dir + "/" + "train_accs.npy", train_accs)
    np.save(policy_eval_dir + "/" + "train_losses.npy", train_losses)
    np.save(policy_eval_dir + "/" + "val_accs.npy", val_accs)
    np.save(policy_eval_dir + "/" + "val_losses.npy", val_losses)
    np.save(policy_eval_dir + "/" + "test_accs.npy", test_accs)
    np.save(policy_eval_dir + "/" + "test_losses.npy", test_losses)
    np.save(policy_eval_dir + "/" + "gt_rewards.npy", gt_rewards)
    np.save(policy_eval_dir + "/" + "learned_rewards.npy", learned_rewards)
    np.save(policy_eval_dir + "/" + "successes.npy", successes)
    if not nn:
        weights = np.asarray(weights)
        np.save(policy_eval_dir + "/" + "weights.npy", weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help="seed")
    parser.add_argument('--num_al_iter', default=0, type=int, help="number of active learning iterations (where 1 is equivalent to normal pref-based reward learning")
    parser.add_argument('--mix', default=None, type=float, help="hyperparameter for how much to mix in new rollouts, where 1 means the next iteration consists of ONLY new rollouts")
    parser.add_argument('--union', default=None, type=int, help="hyperparameter for the number of rollouts from the new policy")
    parser.add_argument('--retrain', dest='retrain', default=False, action='store_true', help="whether to retrain reward and policy from scratch in each active learning iteration")
    parser.add_argument('--nn', dest='nn', default=False, action='store_true', help="whether to use a neural net for reward fn")
    parser.add_argument('--reward_epochs_per_iter', default=100, type=int, help='the number of reward learning epochs to run in one active learning iteration')
    parser.add_argument('--rl_steps_per_iter', default=1000000, type=int, help='the number of RL steps to run in one active learning iteration')


    args = parser.parse_args()

    seed = args.seed
    num_al_iter = args.num_al_iter
    mixing_factor = args.mix
    union_rollouts = args.union
    retrain = args.retrain
    nn = args.nn
    reward_epochs_per_iter = args.reward_epochs_per_iter
    rl_steps_per_iter = args.rl_steps_per_iter

    run_active_learning(num_al_iter, mixing_factor, union_rollouts, retrain, seed, nn, reward_epochs_per_iter, rl_steps_per_iter)




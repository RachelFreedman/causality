import mujoco_gym.learn
import argparse
import numpy as np


EVAL_SEED = 3
EVAL_LEARNED_REWARD = True

# infile is a text file with paths to trained policies
# separated by newline.
def evaluate_policies(infile, outdir):
    with open(infile) as f:
        policy_paths = f.readlines()
    policy_paths = [s.strip() for s in policy_paths]

    gt_reward_means = []
    slearned_reward_means = []
    mlearned_reward_means = []
    llearned_reward_means = []
    success_means = []
    for policy_path in policy_paths:
        if policy_path:
            # 1. Evaluate on the GT reward.
            reward_mean, reward_std, success_mean, success_std = mujoco_gym.learn.evaluate_policy("HalfCheetah-v2", "sac", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False)
            gt_reward_means.append(reward_mean)
            success_means.append(success_mean)

            if EVAL_LEARNED_REWARD:
                # 2. Evaluate on the learned reward(s)
                if 'seed0' in policy_path:
                    seed = 0
                elif 'seed1' in policy_path:
                    seed = 1
                elif 'seed2' in policy_path:
                    seed = 2
                else:
                    raise ValueError("Seed not specified.")
                # sconfig = "vanilla/120demos_allpairs_hdim128-64_100epochs_10patience_00001lr_00001weightdecay"
                # reward_model_path = "/home/jeremy/gym/trex/models/"+sconfig+"_seed"+str(seed)+".params"
                # reward_mean, reward_std, _, _ = mujoco_gym.learn.evaluate_policy("ReacherLearnedReward-v0", "sac", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False, reward_net_path=reward_model_path)
                # slearned_reward_means.append(reward_mean)

                mconfig = "halfcheetah/vanilla/120demos_allpairs_hdim128-64_100epochs_10patience_00001lr_00001weightdecay"
                reward_model_path = "/home/jeremy/gym/trex/models/"+mconfig+"_seed"+str(seed)+".params"
                reward_mean, reward_std, _, _ = mujoco_gym.learn.evaluate_policy("HalfCheetahLearnedReward-v0", "sac", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False, reward_net_path=reward_model_path)
                mlearned_reward_means.append(reward_mean)

                # lconfig = "vanilla/324demos_hdim128-64_stateaction_allpairs_100epochs_10patience_001lr_00001weightdecay"
                # reward_model_path = "/home/jeremy/gym/trex/models/"+lconfig+"_seed"+str(seed)+".params"
                # reward_mean, reward_std, _, _ = mujoco_gym.learn.evaluate_policy("ReacherLearnedReward-v0", "sac", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False, reward_net_path=reward_model_path)
                # llearned_reward_means.append(reward_mean)

    gt_reward_means = np.asarray(gt_reward_means)
    # slearned_reward_means = np.asarray(slearned_reward_means)
    mlearned_reward_means = np.asarray(mlearned_reward_means)
    # llearned_reward_means = np.asarray(llearned_reward_means)
    success_means = np.asarray(success_means)
    np.save(outdir + "/gtrewards.npy", gt_reward_means)
    if EVAL_LEARNED_REWARD:
        # np.save(outdir + "/40demos_hdim128-64_stateaction_allpairs_100epochs_10patience_001lr_00001weightdecay_learnedrewards.npy", slearned_reward_means)
        np.save(outdir + "/120demos_allpairs_hdim128-64_100epochs_10patience_00001lr_00001weightdecay_learnedrewards.npy", mlearned_reward_means)
        # np.save(outdir + "/324demos_hdim128-64_stateaction_allpairs_100epochs_10patience_001lr_00001weightdecay_learnedrewards.npy", llearned_reward_means)
    np.save(outdir + "/success.npy", success_means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--infile', default='',
                        help='Input file with policy paths.')
    parser.add_argument('--outdir', default='',
                        help='Output directory.')

    args = parser.parse_args()

    evaluate_policies(args.infile, args.outdir)

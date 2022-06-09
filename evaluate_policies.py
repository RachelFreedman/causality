import mujoco_gym.learn
import argparse
import numpy as np


EVAL_SEED = 3


# infile is a text file with paths to trained policies
# separated by newline.
def evaluate_policies(infile, outdir):
    with open(infile) as f:
        policy_paths = f.readlines()

    reward_means = []
    success_means = []
    for policy_path in policy_paths:
        reward_mean, reward_std, success_mean, success_std = mujoco_gym.learn.evaluate_policy("Reacher-v2", "sac", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False)
        reward_means.append(reward_mean)
        success_means.append(success_mean)

    reward_means = np.asarray(reward_means)
    success_means = np.asarray(success_means)
    np.save(outdir + "/rewards.npy", reward_means)
    np.save(outdir + "/success.npy", success_means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--infile', default='',
                        help='Input file with policy paths.')
    parser.add_argument('--outdir', default='',
                        help='Output directory.')

    args = parser.parse_args()

    evaluate_policies(args.infile, args.outdir)

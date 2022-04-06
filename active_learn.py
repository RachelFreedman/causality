import trex.model
import mujoco_gym.learn
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_al_iter', default=0, type=int, help="number of active learning iterations")

    args = parser.parse_args()

    num_al_iter = args.num_al_iter

    # Load demonstrations from file and initialize pool of demonstrations

    # For num_al_iter active learning iterations:
    for i in range(num_al_iter):
        # 1. Run reward learning
        # Use the al_data argument to input our pool of changing demonstrations

        # 2. Run RL (using the learned reward)

        # 3. Load RL policy, generate rollouts (number depends on mixing factor), and rank according to GT reward

        # 4. Based on mixing factor, sample demonstrations from previous iteration accordingly
        # Update our pool of demonstrations
        pass




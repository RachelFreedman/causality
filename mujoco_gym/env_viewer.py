import gym, sys, argparse
import numpy as np

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def sample_action(env):
    return env.action_space.sample()

def viewer(env_name):
    env = gym.make(env_name)

    while True:
        done = False
        env.render()
        observation = env.reset()
        action = sample_action(env)
        print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        while not done:
            env.render()  # This env.render() is necessary to render each frame in MuJoCo
            observation, reward, done, info = env.step(sample_action(env))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mujoco Gym Environment Viewer')
    parser.add_argument('--env', default='Reacher-v2',
                        help='Environment to test (default: Reacher-v2)')
    args = parser.parse_args()

    viewer(args.env)

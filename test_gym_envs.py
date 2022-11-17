import gym
env = gym.make('LunarLander-v2')
env.reset()
for _ in range(5000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    print("done", done)
env.close()

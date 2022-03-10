# A Study of Causal Confusion in Preference-Based Reward Learning
Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca D. Dragan, and Daniel Brown

This repository contains the code and data for the **Reacher** preference learning benchmark proposed in the paper. 

See the [project website](https://sites.google.com/view/causal-reward-confusion) for supplemental results and videos.
***

## Installation and Setup
We encourage installing in a python virtualenv or conda environment with Python 3.6 or 3.7.

The setup requires a functional MuJoCo installation. 
A free license can be found [here](https://github.com/openai/mujoco-py).
Follow the instructions at [OpenAI's `mujoco-py` repository](https://github.com/openai/mujoco-py) to install `mujoco_py`. 

After setting up MuJoCo, run the following command in a terminal window to set up a conda environment with all the required dependencies: 
```bash
conda env create --file environment.yml
```

You can visualize the Reacher environment using the environment viewer.  
```bash
python3 -m assistive_gym --env "FeedingSawyer-v1"
```
```bash
python3 -m assistive_gym --env "ScratchItchJaco-v1"
```


## Demonstrations and Pairwise Preference Data
We provide a variety of trajectories and their corresponding rewards for use as demonstrations in preference learning.
Namely, for each environment, we provide:
1. `demos.npy` -- the trajectory data, with shape `(num_trajectories, trajectory_length, observation_dimension)`. (Note: `trajectory_length` is 200 for both Feeding and Itch Scratching.) 
2. `demo_rewards.npy` -- the final cumulative ground truth reward achieved by the corresponding demonstration in `demos.py`. Has shape `(num_trajectories, )`. 
3. `demo_reward_per_timestep.npy` -- the ground truth reward earned by the agent at each timestep in the corresponding demonstration in `demos.npy`. Has shape `(num_trajectories, trajectory_length)`.

The locations of the demonstration data for each environment are:
- Feeding
    - **Raw** Feature-space: 
        - `assistive-gym/trex/data/raw_data/demos_states.npy`
        - `assistive-gym/trex/data/raw_data/demo_rewards.npy`
        - `assistive-gym/trex/data/raw_data/demo_reward_per_timestep.npy`
    - **Augmented** Feature-space: 
        - `assistive-gym/trex/data/augmented_features/demos.npy`
        - `assistive-gym/trex/data/augmented_features/demos_rewards.npy`
        - `assistive-gym/trex/data/augmented_features/demo_reward_per_timestep.npy`
- Itch Scratching
    - **Raw** Feature-space: 
        - `assistive-gym/trex/data/raw_data/demos_states.npy`
        - `assistive-gym/trex/data/raw_data/demo_rewards.npy`
        - `assistive-gym/trex/data/raw_data/demo_reward_per_timestep.npy`
    - **Augmented** Feature-space: 
        - `assistive-gym/trex/data/scratchitch/augmented/demos.npy`
        - `assistive-gym/trex/data/scratchitch/augmented/demos_rewards.npy`
        - `assistive-gym/trex/data/scratchitch/augmented/demo_reward_per_timestep.npy`
        
To load the data into numpy arrays, one can simply run
```python
demos = np.load("##[DEMOS.NPY PATH]##")
demo_rewards = np.load("##[DEMO_REWARDS.NPY PATH]##")
demo_reward_per_timestep = np.load("##[DEMO_REWARD_PER_TIMESTEP.NPY PATH]##")
```
(where `##[DEMOS.NPY PATH]##` is a path to a `demos.npy` file listed above) within a Python script. 


## Reward Learning from Preferences
We provide `trex/model.py`, a convenient script that loads the trajectory data, creates the pairwise preferences based on the ground truth reward, and performs reward learning on the pairwise preferences. 
To perform reward learning for each of the benchmark environments, run the following in the `assistive-gym/` directory:
- Feeding
    ```bash
    python3 trex/model.py --hidden_dims 128 64 --num_comps 2000 --num_epochs 100 --patience 10 --lr 0.01 --weight_decay 0.01 --seed 0 --reward_model_path ./reward_models/model.params
    ```
- Itch Scratching
    ```bash
    python3 trex/model.py --scratch_itch --hidden_dims 128 64 --num_comps 2000 --num_epochs 100 --patience 10 --lr 0.01 --weight_decay 0.01 --seed $seed --reward_model_path ./reward_models/model.params
    ```
The trained parameters of the reward network will be saved in `assistive-gym/reward_models/model.params`.


## Training the RL Policy
Once the reward network is trained, we can perform reinforcement learning using the preference-learned reward. 
To train, run:
- Feeding
    ```bash
    python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --algo ppo --seed $seed --train --train-timesteps 1000000 --reward-net-path ./reward_models/model.params --save-dir ./trained_policies/
    ```
- Itch Scratching
    ```bash
    python3 -m assistive_gym.learn --env "ScratchItchLearnedRewardJaco-v0" --algo ppo --seed $seed --train --train-timesteps 1000000 --reward-net-path $reward_model_path --save-dir ./trained_policies/
    ```
 
To evaluate the trained policy on 100 rollouts using the ground truth reward:
- Feeding
    ```bash
      python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path ./trained_policies/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_000053/checkpoint-53
    ```
- Itch Scratching
    ```bash
      python3 -m assistive_gym.learn --env "ScratchItchJaco-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path ./trained_policies/ppo/ScratchItchLearnedRewardJaco-v0/checkpoint_000053/checkpoint-53
    ```

And to render rollouts of the trained policy:
- Feeding
    ```bash
      python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --render --render-episodes 3 --seed 3 --load-policy-path ./trained_policies/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_000053/checkpoint-53
    ```
- Itch Scratching
    ```bash
      python3 -m assistive_gym.learn --env "ScratchItchJaco-v1" --algo ppo --render --render-episodes 3 --seed 3 --load-policy-path ./trained_policies/ppo/ScratchItchLearnedRewardJaco-v0/checkpoint_000053/checkpoint-53
    ```


## Gym

Gym is an open source Python library for developing and comparing reinforcement learning algorithms by providing a standard API to communicate between learning algorithms and environments, as well as a standard set of environments compliant with that API. Since its release, Gym's API has become the field standard for doing this.

Gym currently has two pieces of documentation: the [documentation website](http://gym.openai.com) and the [FAQ](https://github.com/openai/gym/wiki/FAQ).

## Installation

To install the base Gym library, use `pip install gym`.

This does not include dependencies for all families of environments (there's a massive number, and some can be problematic to install on certain systems). You can install these dependencies for one family like `pip install gym[atari]` or use `pip install gym[all]` to install all dependencies.

We support Python 3.7, 3.8, 3.9 and 3.10 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

## API

The Gym API's API models environments as simple Python `env` classes. Creating environment instances and interacting with them is very simple- here's an example using the "CartPole-v1" environment:

```python
import gym 
env = gym.make('CartPole-v1')

# env is created, now we can use it: 
for episode in range(10): 
    observation = env.reset()
    for step in range(50):
        action = env.action_space.sample()  # or given a custom model, action = policy(observation)
        observation, reward, done, info = env.step(action)
```

## Notable Related Libraries

* [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) is a learning library based on the Gym API. It is our recommendation for beginners who want to start learning things quickly.
* [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) builds upon SB3, containing optimal hyperparameters for Gym environments as well as code to easily find new ones. Such tuning is almost always required.
* The [Autonomous Learning Library](https://github.com/cpnota/autonomous-learning-library) and [Tianshou](https://github.com/thu-ml/tianshou) are two reinforcement learning libraries I like that are generally geared towards more experienced users.
* [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) is like Gym, but for environments with multiple agents.

## Environment Versioning

Gym keeps strict versioning for reproducibility reasons. All environments end in a suffix like "\_v0".  When changes are made to environments that might impact learning results, the number is increased by one to prevent potential confusion.

## Citation

A whitepaper from when Gym just came out is available https://arxiv.org/pdf/1606.01540, and can be cited with the following bibtex entry:

```
@misc{1606.01540,
  Author = {Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba},
  Title = {OpenAI Gym},
  Year = {2016},
  Eprint = {arXiv:1606.01540},
}
```

## Release Notes

There used to be release notes for all the new Gym versions here. New release notes are being moved to [releases page](https://github.com/openai/gym/releases) on GitHub, like most other libraries do. Old notes can be viewed [here](https://github.com/openai/gym/blob/31be35ecd460f670f0c4b653a14c9996b7facc6c/README.rst).

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
python3 mujoco_gym/env_viewer --env "Reacher-v2"
```

## Demonstrations and Pairwise Preference Data
We provide a variety of trajectories and their corresponding rewards for use as demonstrations in preference learning.
Namely, for each environment, we provide:
1. `demos.npy` -- the trajectory data, with shape `(num_trajectories, trajectory_length, observation_dimension)`. (Note: `trajectory_length` is 50 for Reacher.) 
2. `demo_rewards.npy` -- the final cumulative ground truth reward achieved by the corresponding demonstration in `demos.py`. Has shape `(num_trajectories, )`. 
3. `demo_reward_per_timestep.npy` -- the ground truth reward earned by the agent at each timestep in the corresponding demonstration in `demos.npy`. Has shape `(num_trajectories, trajectory_length)`.

The locations of the demonstration data for each environment are:
- **Raw** Feature-space: 
    - `gym/trex/data/raw/demos.npy`
    - `gym/trex/data/raw/demo_rewards.npy`
    - `gym/trex/data/raw/demo_reward_per_timestep.npy`
- **Augmented** Feature-space: 
    - `gym/trex/data/augmented/demos.npy`
    - `gym/trex/data/augmented/demos_rewards.npy`
    - `gym/trex/data/augmented/demo_reward_per_timestep.npy`
        
To load the data into numpy arrays, one can simply run
```python
demos = np.load("##[DEMOS.NPY PATH]##")
demo_rewards = np.load("##[DEMO_REWARDS.NPY PATH]##")
demo_reward_per_timestep = np.load("##[DEMO_REWARD_PER_TIMESTEP.NPY PATH]##")
```
(where `##[DEMOS.NPY PATH]##` is a path to a `demos.npy` file listed above) within a Python script. 


## Reward Learning from Preferences
We provide `trex/model.py`, a convenient script that loads the trajectory data, creates the pairwise preferences based on the ground truth reward, and performs reward learning on the pairwise preferences. 
To perform reward learning for each of the benchmark environments, run the following in the `gym/` directory:
```bash
python3 trex/model.py --hidden_dims 128 64 --num_comps 2000 --num_epochs 100 --patience 10 --lr 0.01 --weight_decay 0.01 --seed 0 --reward_model_path ./reward_models/model.params
```
The trained parameters of the reward network will be saved in `gym/reward_models/model.params`.


## Training the RL Policy
Once the reward network is trained, we can perform reinforcement learning using the preference-learned reward. 
To train, run:
```bash
python3 mujoco_gym/learn.py --env "ReacherLearnedReward-v0" --algo sac --seed 0 --train --train-timesteps 1000000 --reward-net-path ./reward_models/model.params --save-dir ./trained_policies/
```
 
To evaluate the trained policy on 100 rollouts using the ground truth reward:
```bash
  python3 mujoco_gym/learn.py --env "Reacher-v2" --algo sac --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path ./trained_policies/ppo/ReacherLearnedReward-v0/checkpoint_002231/checkpoint-2231
```

And to render rollouts of the trained policy:
```bash
  python3 mujoco_gym/learn.py --env "Reacher-v2" --algo sac --render --render-episodes 3 --seed 3 --load-policy-path ./trained_policies/ppo/ReacherLearnedReward-v0/checkpoint_002231/checkpoint-2231
```

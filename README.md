# Causal Confusion and Reward Misidentification in Preference-Based Reward Learning
_Jeremy Tien, Jerry Zhi-Yang He, Zackory Erickson, Anca D. Dragan, and Daniel S. Brown_

This repository contains the code and data for the **Reacher**, **Half Cheetah**, and **Lunar Lander** preference learning benchmark environments presented in [**"Causal Confusion and Reward Misidentification in Preference-Based Reward Learning"**](https://openreview.net/pdf?id=R0Xxvr_X3ZA) (ICLR 2023). 

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
python3 mujoco_gym/env_viewer.py --env "Reacher-v2"
```
Replace `Reacher` with `HalfCheetah` or `LunarLander` to visualize the Half Cheetah and Lunar Lander environments, respectively. 

## Demonstrations and Pairwise Preference Data
We provide a variety of trajectories and their corresponding rewards for use as demonstrations in preference learning.
Namely, we provide:
1. `demos.npy` -- the trajectory data, with shape `(num_trajectories, trajectory_length, observation_dimension)`. (Note: `trajectory_length` is 50 for Reacher.) 
2. `demo_rewards.npy` -- the final cumulative ground truth reward achieved by the corresponding demonstration in `demos.py`. Has shape `(num_trajectories, )`. 
3. `demo_reward_per_timestep.npy` -- the ground truth reward earned by the agent at each timestep in the corresponding demonstration in `demos.npy`. Has shape `(num_trajectories, trajectory_length)`.

The locations of the demonstration data for each environment are:
- Reacher
    - "**Full**" Feature-space (default observation features + add'l. features to make ground-truth reward, TRUE, fully-inferrable): 
        - `gym/trex/data/reacher/raw_stateaction/raw_360/demos.npy`
        - `gym/trex/data/reacher/raw_stateaction/raw_360/demo_rewards.npy`
        - `gym/trex/data/reacher/raw_stateaction/raw_360/demo_reward_per_timestep.npy`
    - "**Pure**" Feature-space ("Full" but with distractor features that are not causal wrt. TRUE removed): 
        - `gym/trex/data/reacher/pure_fully_observable/demos.npy`
        - `gym/trex/data/reacher/pure_fully_observable/demo_rewards.npy`
- Half Cheetah
    - "**Full**" Feature-space (default observation features + add'l. features to make ground-truth reward, TRUE, fully-inferrable): 
        - `gym/trex/data/halfcheetah/raw_stateaction/demos.npy`
        - `gym/trex/data/halfcheetah/raw_stateaction/demo_rewards.npy`
        - `gym/trex/data/halfcheetah/raw_stateaction/demo_reward_per_timestep.npy`
- Lunar Lander
    - "**Full**" Feature-space (default observation features + add'l. features to make ground-truth reward, TRUE, fully-inferrable): 
        - `gym/trex/data/lunarlander/raw_stateaction/demos.npy`
        - `gym/trex/data/lunarlander/raw_stateaction/demo_rewards.npy`
        - `gym/trex/data/lunarlander/raw_stateaction/demo_reward_per_timestep.npy`
        
To load the data into numpy arrays, one can simply run
```python
demos = np.load("##[DEMOS.NPY PATH]##")
demo_rewards = np.load("##[DEMO_REWARDS.NPY PATH]##")
demo_reward_per_timestep = np.load("##[DEMO_REWARD_PER_TIMESTEP.NPY PATH]##")
```
(where `##[DEMOS.NPY PATH]##` is a path to a `demos.npy` file listed above) within a Python script. 


## Reward Learning from Preferences
We provide `trex/model.py`, a convenient script that loads the trajectory data, creates the pairwise preferences based on the ground truth reward, and performs reward learning on the pairwise preferences. 
To perform reward learning for each of the benchmark environments (and to replicate our **_Section 4: Evidence of Causal Confusion results_**), run the following in the `gym/` directory:
```bash
python3 trex/model.py --env ${ENV_NAME}-v2 --num_demos ${NUM_DEMOS} --seed 0 --state_action --hidden_dims 128 64 --all_pairs --num_epochs 100 --patience 10 --lr 0.0001 --weight_decay 0.0001 --reward_model_path ./reward_models/model.params
```
where **`${ENV_NAME}`** is one of **`Reacher`**, **`HalfCheetah`**, or **`LunarLander`**, and **${NUM_DEMOS}** is **40**, **120**, or **324** (which correspond to the **S**, **M**, and **L** dataset sizes, respectively). 
The trained parameters of the reward network will be saved in `gym/reward_models/model.params`.


## Training the RL Policy
Once the reward network is trained, we can perform reinforcement learning using the reward learned from preferences. 
To train, run:
```bash
python3 mujoco_gym/learn.py --env "${ENV_NAME}LearnedReward-v0" --algo sac --seed 0 --train --train-timesteps 1000000 --reward-net-path ./reward_models/model.params --save-dir ./trained_policies/
```
 
To evaluate the trained policy on 100 rollouts using the ground truth reward:
```bash
  python3 mujoco_gym/learn.py --env "${ENV_NAME}-v2" --algo sac --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path ./trained_policies/sac/${ENV_NAME}LearnedReward-v0/checkpoint_002231/checkpoint-2231
```

And to render rollouts of the trained policy:
```bash
  python3 mujoco_gym/learn.py --env "${ENV_NAME}-v2" --algo sac --render --render-episodes 3 --seed 3 --load-policy-path ./trained_policies/sac/${ENV_NAME}LearnedReward-v0/checkpoint_002231/checkpoint-2231
```
where `${ENV_NAME}` is again replaced with the name of the desired environment.

## Citation
If you use the benchmark data and/or scripts, please cite:
```bibtex
@inproceedings{tien2023causal,
    title={Causal Confusion and Reward Misidentification in Preference-Based Reward Learning},
    author={Jeremy Tien and Jerry Zhi-Yang He and Zackory Erickson and Anca Dragan and Daniel S. Brown},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=R0Xxvr_X3ZA}
}
```

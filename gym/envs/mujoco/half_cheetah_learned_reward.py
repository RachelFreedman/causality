import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch

from .half_cheetah import HalfCheetahEnv
from trex.model import Net
from gpu_utils import determine_default_torch_device


class HalfCheetahLearnedRewardEnv(HalfCheetahEnv):
    def __init__(self, reward_net_path, indvar=None):
        # super(HalfCheetahLearnedRewardEnv, self).__init__()

        # Reward Model Specifications
        self.augmented = False
        self.augmented_full = False
        self.pure_fully_observable = False
        self.num_rawfeatures = 17  # indvar[0]  # HalfCheetah has 17 raw observation features total
        self.num_distractorfeatures = None  # Not defined yet
        self.state_action = True
        self.hidden_dims = (128, 64)
        self.normalize = False

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
        self.reward_net = Net(env_name="HalfCheetah-v2", hidden_dims=self.hidden_dims, augmented=self.augmented, augmented_full=self.augmented_full, num_rawfeatures=self.num_rawfeatures, num_distractorfeatures=self.num_distractorfeatures, state_action=self.state_action, pure_fully_observable=self.pure_fully_observable, norm=self.normalize)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

        super(HalfCheetahLearnedRewardEnv, self).__init__()

    def step(self, a):
        obs, reward, done, info = super().step(a)

        # Store the ground-truth reward for downstream use (but not training).
        info['gt_reward'] = reward

        if self.pure_fully_observable:
            raise Exception("Not implemented yet.")
        elif self.augmented_full:
            raise Exception("Not implemented yet.")
        elif self.augmented:
            raise Exception("Not implemented yet.")
        elif self.state_action:
            input = np.concatenate((obs, a))
        else:
            input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info


import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch

from .reacher import ReacherEnv
from trex.model import Net
from gpu_utils import determine_default_torch_device


class ReacherLearnedRewardEnv(ReacherEnv):
    def __init__(self, reward_net_path, indvar=None):
        # super(ReacherLearnedRewardEnv, self).__init__()

        # Reward Model Specifications
        self.augmented_full = True
        self.augmented = False
        self.num_rawfeatures = 11  # Reacher has 11 raw features total
        self.hidden_dims = tuple()
        self.normalize = False

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
        self.reward_net = Net(hidden_dims=self.hidden_dims, augmented=self.augmented, augmented_full=self.augmented_full, num_rawfeatures=self.num_rawfeatures, norm=self.normalize)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

        super(ReacherLearnedRewardEnv, self).__init__()

    def step(self, a):
        obs, reward, done, info = super().step(a)

        distance = np.linalg.norm(obs[8:11])
        action_norm = np.linalg.norm(a)
        privileged_features = np.array([distance, action_norm])

        if self.augmented_full:
            input = np.concatenate((obs[0:self.num_rawfeatures], privileged_features))
        elif self.augmented:
            input = np.concatenate((obs[0:self.num_rawfeatures], [privileged_features[0]]))
        else:
            input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info


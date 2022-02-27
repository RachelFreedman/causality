import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch

from .reacher import ReacherEnv
from trex.model import Net


class ReacherLearnedRewardEnv(ReacherEnv):
    def __init__(self, reward_net_path, indvar=None):
        # super(ReacherLearnedRewardEnv, self).__init__()

        # Reward Model Specifications
        self.augmented = True
        self.num_rawfeatures = indvar[0]  # Reacher has 11 raw features total
        self.hidden_dims = tuple()

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net = Net(hidden_dims=self.hidden_dims, augmented=self.augmented, num_rawfeatures=self.num_rawfeatures)
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

        super(ReacherLearnedRewardEnv, self).__init__()

    def step(self, a):
        obs, reward, done, info = super().step(a)

        distance = np.linalg.norm(obs[8:11])
        handpicked_features = np.array([distance])

        if self.augmented:
            input = np.concatenate((obs[0:self.num_rawfeatures], handpicked_features))
        else:
            input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info


import numpy as np
from gym import utils
# from gym.envs.box2d import box2d
import torch

#import sys
#sys.path.insert(0, '/home/jeremy/gym')

from .lunar_lander import LunarLander
from trex.model import Net
from discriminator_kl import Discriminator
from gpu_utils import determine_default_torch_device


class LunarLearnedReward(LunarLander):
    def __init__(self, reward_net_path, indvar=None):

        # Reward Model Specifications
        self.pure_fully_observable = False
        self.state_action = True
        self.hidden_dims = (128, 64)
        self.normalize = False
        self.kl_penalty = 0.0
        self.discriminator_net_path = ""

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())

        self.reward_net = Net(env_name="LunarLander-v2", hidden_dims=self.hidden_dims, state_action=self.state_action, pure_fully_observable=self.pure_fully_observable, norm=self.normalize)
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

        if self.kl_penalty > 0.0:
            self.discriminator_net = Discriminator(env_name="LunarLander-v2", hidden_dims=(128, 128, 128), fully_observable=self.state_action, pure_fully_observable=self.pure_fully_observable, norm=False)
            self.discriminator_net.load_state_dict(torch.load(self.discriminator_net_path, map_location=torch.device('cpu')))
            self.discriminator_net.to(self.device)

        super(LunarLearnedReward, self).__init__()

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Store the ground-truth reward for downstream use (but not training).
        info['gt_reward'] = reward

        if self.pure_fully_observable:
            raise NotImplementedError("pure_fully_observable not implemented for LunarLander yet!")
        elif self.state_action:
            input = np.concatenate((obs, action))
        else:
            input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()
            if self.kl_penalty > 0.0:  # The KL divergence penalty
                reward -= self.kl_penalty * self.discriminator_net.forward(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info


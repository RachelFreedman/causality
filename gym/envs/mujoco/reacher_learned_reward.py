import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import torch

#import sys
#sys.path.insert(0, '/home/jeremy/gym')

from .reacher import ReacherEnv
from trex.model import Net
from discriminator_kl import Discriminator
from gpu_utils import determine_default_torch_device


class ReacherLearnedRewardEnv(ReacherEnv):
    def __init__(self, reward_net_path, indvar=None):
        # super(ReacherLearnedRewardEnv, self).__init__()

        # Reward Model Specifications
        self.pure_fully_observable = False
        self.augmented_full = False
        self.augmented = False
        self.num_rawfeatures = 11  # indvar[0]  # Reacher has 11 raw features total
        self.num_distractorfeatures = 8
        self.state_action = True
        self.hidden_dims = (128, 64)
        self.normalize = False
        self.kl_penalty = indvar[0]
        self.discriminator_net_path = "/home/jeremy/gym/discriminator_kl_models/reacher/vanilla/" + reward_net_path[reward_net_path.index("vanilla")+8:]

        print("reward_net_path:", reward_net_path)
        self.reward_net_path = reward_net_path

        self.device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
        print("device:", self.device)
        print("torch.cuda.is_available():", torch.cuda.is_available())

        self.reward_net = Net(env_name="Reacher-v2", hidden_dims=self.hidden_dims, augmented=self.augmented, augmented_full=self.augmented_full, num_rawfeatures=self.num_rawfeatures, num_distractorfeatures=self.num_distractorfeatures, state_action=self.state_action, pure_fully_observable=self.pure_fully_observable, norm=self.normalize)
        self.reward_net.load_state_dict(torch.load(self.reward_net_path, map_location=torch.device('cpu')))
        self.reward_net.to(self.device)

        if self.kl_penalty > 0.0:
            self.discriminator_net = Discriminator(env_name="Reacher-v2", hidden_dims=(128, 128, 128), fully_observable=self.state_action, pure_fully_observable=self.pure_fully_observable, norm=False)
            self.discriminator_net.load_state_dict(torch.load(self.discriminator_net_path, map_location=torch.device('cpu')))
            self.discriminator_net.to(self.device)

        super(ReacherLearnedRewardEnv, self).__init__()

    def step(self, a):
        obs, reward, done, info = super().step(a)

        # Store the ground-truth reward for downstream use (but not training).
        info['gt_reward'] = reward

        distance = np.linalg.norm(obs[8:11])
        action_norm = np.linalg.norm(a)
        privileged_features = np.array([distance, action_norm])

        if self.pure_fully_observable:
            input = np.concatenate((obs[8:11], a))
        elif self.augmented_full:
            input = np.concatenate((obs[0:self.num_rawfeatures], privileged_features))
        elif self.augmented:
            input = np.concatenate((obs[0:self.num_rawfeatures], [privileged_features[0]]))
        else:
            if self.state_action:
                input = np.concatenate((obs[0:self.num_distractorfeatures], obs[8:11], a))
            else:
                input = obs

        # Just modify the reward
        with torch.no_grad():
            reward = self.reward_net.cum_return(torch.from_numpy(np.array([input])).float().to(self.device)).item()
            if self.kl_penalty > 0.0:  # The KL divergence penalty
                reward -= self.kl_penalty * self.discriminator_net.forward(torch.from_numpy(np.array([input])).float().to(self.device)).item()

        return obs, reward, done, info


from os.path import exists
import trex.model
import mujoco_gym.learn
import argparse
import numpy as np
import multiprocessing, ray
import re, string
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from gpu_utils import determine_default_torch_device


# Function similar to active_learn's get_rollouts to get rollouts from trained policy
# env_name: HalfCheetah-v2 or Reacher-v2
def get_rollouts(env_name, num_rollouts, policy_path, seed, pure_fully_observable=False, fully_observable=False):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    # Set up the environment
    env = mujoco_gym.learn.make_env(env_name, seed=seed)
    # Load pretrained policy from file
    test_agent, _ = mujoco_gym.learn.load_policy(env, 'sac', env_name, policy_path, seed=seed)

    new_rollouts = []
    new_rollout_rewards = []
    for r in range(num_rollouts):
        traj = []
        reward_total = 0.0
        obs = env.reset()
        info = None
        done = False
        while not done:
            action = test_agent.compute_action(obs)

            # Reacher privileged features: end effector - target distance
            if env_name == "Reacher-v2":
                distance = np.linalg.norm(obs[8:11])
                action_norm = np.linalg.norm(action)
                privileged_features = np.array([distance, action_norm])
                pure_obs = obs[8:11]
            elif env_name == "HalfCheetah-v2":
                pass

            if pure_fully_observable:
                data = np.concatenate((pure_obs, action))
            elif fully_observable:
                data = np.concatenate((obs, action))
            else:
                data = obs

            obs, reward, done, info = env.step(action)

            traj.append(data)
            reward_total += reward

        new_rollouts.append(traj)
        new_rollout_rewards.append(reward_total)

    new_rollouts = np.asarray(new_rollouts)
    new_rollout_rewards = np.asarray(new_rollout_rewards)
    return new_rollouts


# Function that prepares the training data:
# - Squishes the rollouts / trajectories into just a bunch of states
# - Adds labels of 0 for reward-learning and 1 for policy training.
# Dimensions of reward_learning_trajs and policy_trajs: (num_trajs, traj_length, feature_dim)
def prepare_data(reward_learning_trajs, policy_trajs):
    reward_learning_obs = np.reshape(reward_learning_trajs, (reward_learning_trajs.shape[0] * reward_learning_trajs.shape[1], reward_learning_trajs.shape[2]))
    policy_obs = np.reshape(policy_trajs, (policy_trajs.shape[0] * policy_trajs.shape[1], policy_trajs.shape[2]))
    reward_learning_labels = np.zeros((reward_learning_obs.shape[0], 1))
    policy_labels = np.ones((policy_obs.shape[0], 1))
    reward_learning_data = np.concatenate((reward_learning_obs, reward_learning_labels), axis=-1)
    policy_data = np.concatenate((policy_obs, policy_labels), axis=-1)

    data = np.concatenate((reward_learning_data, policy_data), axis=0)
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]


# Define discriminator model that classifies between states from reward-learning training data (0)
# and states from the trained policy (1) by minimizing cross-entropy loss.
class Discriminator(nn.Module):
    def __init__(self, env_name, hidden_dims=(128, 64), fully_observable=False, pure_fully_observable=False, norm=False):
        super().__init__()

        if pure_fully_observable:
            if env_name == "Reacher-v2":
                input_dim = 5
            elif env_name == "HalfCheetah-v2":
                raise Exception("NOT IMPLEMENTED YET.")
        elif fully_observable:
            if env_name == "Reacher-v2":
                input_dim = 13
            elif env_name == "HalfCheetah-v2":
                raise Exception("NOT IMPLEMENTED YET.")
        else:
            if env_name == "Reacher-v2":
                input_dim = 13
            elif env_name == "HalfCheetah-v2":
                raise Exception("NOT IMPLEMENTED YET.")

        self.normalize = norm
        if self.normalize:
            print("Normalizing input features...")
            self.layer_norm = nn.LayerNorm(input_dim)
        self.num_layers = len(hidden_dims) + 1

        self.fcs = nn.ModuleList([None for _ in range(self.num_layers)])
        if len(hidden_dims) == 0:
            self.fcs[0] = nn.Linear(input_dim, 1, bias=False)
        else:
            self.fcs[0] = nn.Linear(input_dim, hidden_dims[0])
            for l in range(len(hidden_dims)-1):
                self.fcs[l+1] = nn.Linear(hidden_dims[l], hidden_dims[l+1])
            self.fcs[len(hidden_dims)] = nn.Linear(hidden_dims[-1], 1, bias=False)

        print(self.fcs)

    def forward(self, x):
        # Normalize features
        if self.normalize:
            x = self.layer_norm(x)

        for l in range(self.num_layers - 1):
            x = F.leaky_relu(self.fcs[l](x))
        logit = self.fcs[-1](x)
        return logit


def train(device, discriminator_network, optimizer, training_inputs, training_outputs, num_epochs, l1_reg, checkpoint_dir, val_obs, val_labels, patience, return_weights=False):
    print("device:", device)
    # Note that a sigmoid is implicitly applied in the BCEWithLogitsLoss
    loss_criterion = nn.BCEWithLogitsLoss()

    trigger_times = 0
    prev_min_val_loss = 100
    training_data = list(zip(training_inputs, training_outputs))
    final_weights = None
    for epoch in range(num_epochs):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        # for i in range(len(training_labels)):
        #     obs = np.array(training_obs[i])
        #     label = np.array([training_labels[i]])
        #     obs = torch.from_numpy(obs).float().to(device)
        #     label = torch.from_numpy(label).to(device)
        #
        #     # zero out gradient
        #     optimizer.zero_grad()
        #
        #     # forward + backward + optimize
        #     outputs = discriminator_network.forward(obs)
        #     print("outputs:", outputs.shape)
        #     print("label")
        #     # outputs = outputs.unsqueeze(0)
        #     # print("train outputs", outputs.shape)
        #     # print("train label", label.shape)
        #
        #     # Calculate loss
        #     cross_entropy_loss = loss_criterion(outputs, label)
        #     l1_loss = l1_reg * torch.linalg.vector_norm(torch.cat([param.view(-1) for param in discriminator_network.parameters()]), 1)
        #     loss = cross_entropy_loss + l1_loss
        #
        #     # Backpropagate
        #     loss.backward()
        #
        #     # Take one optimizer step
        #     optimizer.step()
        training_obs = np.array(training_obs)
        training_labels = np.array(training_labels)
        training_obs = torch.from_numpy(training_obs).float().to(device)
        training_labels = torch.from_numpy(training_labels).to(device)

        # zero out gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = torch.flatten(discriminator_network.forward(training_obs))
        # print("outputs:", outputs.shape)
        # print(outputs)
        # print("training_labels:", training_labels.shape)
        # print(training_labels)
        # outputs = outputs.unsqueeze(0)
        # print("train outputs", outputs.shape)
        # print("train label", label.shape)

        # Calculate loss
        cross_entropy_loss = loss_criterion(outputs, training_labels)
        l1_loss = l1_reg * torch.linalg.vector_norm(torch.cat([param.view(-1) for param in discriminator_network.parameters()]), 1)
        loss = cross_entropy_loss + l1_loss

        # Backpropagate
        loss.backward()

        # Take one optimizer step
        optimizer.step()

        val_loss = calc_val_loss(device, discriminator_network, val_obs, val_labels)
        val_acc = calc_accuracy(device, discriminator_network, val_obs, val_labels)
        print("end of epoch {}: val_loss {}, val_acc {}".format(epoch, val_loss, val_acc))

        # Early Stopping
        if val_loss > prev_min_val_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            if trigger_times >= patience:
                print("Early stopping.")
                if return_weights:
                    return final_weights
                return None
        else:
            trigger_times = 0
            print('trigger times:', trigger_times)
            print("saving model weights...")
            if checkpoint_dir:
                torch.save(discriminator_network.state_dict(), checkpoint_dir)
            if return_weights:
                print("Weights:", discriminator_network.state_dict())
                final_weights = discriminator_network.state_dict()

        prev_min_val_loss = min(prev_min_val_loss, val_loss)
    print("Finished training.")
    if return_weights:
        return final_weights
    return None


# Calculates the cross-entropy losses over the entire validation set and returns the MEAN.
def calc_val_loss(device, discriminator_network, training_inputs, training_outputs):
    loss_criterion = nn.BCEWithLogitsLoss()
    losses = []
    with torch.no_grad():
        for i in range(len(training_inputs)):
            obs = np.array(training_inputs[i])
            label = np.array([training_outputs[i]])
            obs = torch.from_numpy(obs).float().to(device)
            label = torch.from_numpy(label).to(device)

            # Forward to get logits
            outputs = discriminator_network.forward(obs)
            # outputs = outputs.unsqueeze(0)
            # print("val outputs", outputs.shape)
            # print("val label", label.shape)

            loss = loss_criterion(outputs, label)
            losses.append(loss.item())

    return np.mean(losses)


def calc_accuracy(device, discriminator_network, training_inputs, training_outputs):
    with torch.no_grad():
        obs = np.array(training_inputs)
        label = np.array(training_outputs)
        obs = torch.from_numpy(obs).float().to(device)

        # Forward to get logits
        outputs = discriminator_network.forward(obs)
        preds = (outputs > 0).detach().cpu().numpy().flatten()
        # print("preds:", preds)
        # print(preds == label)
        # print(np.sum(preds == label))
        # print(len(label))

    return np.sum(preds == label) / len(label)


def get_logit(device, net, x):
    with torch.no_grad():
        logit = net.forward(torch.from_numpy(x).float().to(device))
    return logit.detach().cpu().numpy().flatten()


def run(env_name, seed, reward_learning_data_path, trained_policy_path, num_trajs, fully_observable, pure_fully_observable,
        load_weights, discriminator_model_path, num_epochs, hidden_dims, lr, weight_decay, l1_reg, patience):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data used in reward-learning
    print("Pulling " + str(num_trajs) + " trajectories from the reward learning training dataset...")
    reward_learning_trajs = np.load(reward_learning_data_path)
    idx = np.round(np.linspace(0, reward_learning_trajs.shape[0] - 1, num_trajs)).astype(int)
    reward_learning_trajs = reward_learning_trajs[idx]

    # Get rollouts from trained policy
    print("Getting " + str(num_trajs) + " rollouts from the trained policy...")
    policy_trajs = get_rollouts(env_name=env_name, num_rollouts=num_trajs, policy_path=trained_policy_path, seed=seed,
                                pure_fully_observable=pure_fully_observable, fully_observable=fully_observable)

    # Create validation set (disjoint set of trajectories)
    idx = np.random.permutation(np.arange(reward_learning_trajs.shape[0]))
    shuffled_reward_learning_trajs = reward_learning_trajs[idx]
    train_val_split_i = int(reward_learning_trajs.shape[0] * 0.5)
    val_reward_learning_trajs = shuffled_reward_learning_trajs[0:train_val_split_i]
    train_reward_learning_trajs = shuffled_reward_learning_trajs[train_val_split_i:]

    idx = np.random.permutation(np.arange(policy_trajs.shape[0]))
    shuffled_policy_trajs = policy_trajs[idx]
    train_val_split_i = int(policy_trajs.shape[0] * 0.5)
    val_policy_trajs = shuffled_policy_trajs[0:train_val_split_i]
    train_policy_trajs = shuffled_policy_trajs[train_val_split_i:]

    # Prepare discriminator training data
    train_obs, train_labels = prepare_data(train_reward_learning_trajs, train_policy_trajs)
    val_obs, val_labels = prepare_data(val_reward_learning_trajs, val_policy_trajs)

    # Train discriminator
    device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
    discriminator_model = Discriminator(env_name=env_name, hidden_dims=hidden_dims,
                                        pure_fully_observable=pure_fully_observable, fully_observable=fully_observable)
    discriminator_model.to(device)

    # Check if we already trained this model before. If so, load the saved weights.
    model_exists = False
    if load_weights:
        model_exists = exists(discriminator_model_path)
        if model_exists:
            print("Found existing model weights! Loading state dict...")
            discriminator_model.load_state_dict(
                torch.load(discriminator_model_path))  # map_location=torch.device('cpu') may be necessary
        else:
            print("Could not find existing model weights. Training from scratch...")
    else:
        print("Training reward model from scratch...")
    if not model_exists:
        num_total_params = sum(p.numel() for p in discriminator_model.parameters())
        num_trainable_params = sum(p.numel() for p in discriminator_model.parameters() if p.requires_grad)
        print("Total number of parameters:", num_total_params)
        print("Number of trainable parameters:", num_trainable_params)

        import torch.optim as optim
        optimizer = optim.Adam(discriminator_model.parameters(), lr=lr, weight_decay=weight_decay)
        train(device, discriminator_model, optimizer, train_obs, train_labels, num_epochs, l1_reg,
              discriminator_model_path, val_obs, val_labels, patience)

    # Report training and validation accuracy
    train_acc = calc_accuracy(device, discriminator_model, train_obs, train_labels)
    val_acc = calc_accuracy(device, discriminator_model, val_obs, val_labels)
    print("train accuracy:", train_acc)
    print("validation accuracy:", val_acc)

    # Inference:
    # For the states visited during reward-learning, calculate the average return/logit. --> $D_{KL}(p(x) || q(x))$
    reward_learning_obs = np.reshape(reward_learning_trajs, (
    reward_learning_trajs.shape[0] * reward_learning_trajs.shape[1], reward_learning_trajs.shape[2]))
    # TODO: check that we are able to take the np.mean -- may need to convert tensor to numpy or vice versa
    reward_learning_logits = get_logit(device, discriminator_model, reward_learning_obs)
    # print("reward_learning_logits:", reward_learning_logits)
    dkl_pq = np.mean(reward_learning_logits) * -1

    # For the states visited during the policy rollout, calculate the average return/logit and NEGATE. --> $D_{KL}(q(x) || p(x))$
    policy_obs = np.reshape(policy_trajs, (policy_trajs.shape[0] * policy_trajs.shape[1], policy_trajs.shape[2]))
    policy_logits = get_logit(device, discriminator_model, policy_obs)
    dkl_qp = np.mean(policy_logits)

    print("dkl (reward learning) || (policy rollouts):", dkl_pq)
    print("dkl (policy rollouts) || (reward learning):", dkl_qp)

    return train_acc, val_acc, dkl_pq, dkl_qp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='', help="")
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    # Have an argument for the reward-learning training data file
    parser.add_argument('--reward_learning_data_path', default='', help="")
    # Have an argument for the trained policy to pull rollouts from
    parser.add_argument('--trained_policy_path', default='', help="")
    # Have an argument for how many trajectories / rollouts to pull from both data pools
    parser.add_argument('--num_trajs', default=100, type=int, help="")
    parser.add_argument('--fully_observable', dest='fully_observable', default=False, action='store_true', help="")
    parser.add_argument('--pure_fully_observable', dest='pure_fully_observable', default=False, action='store_true', help="")
    parser.add_argument('--load_weights', dest='load_weights', default=False, action='store_true', help="")
    parser.add_argument('--discriminator_model_path', default='', help="")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epochs")
    parser.add_argument('--hidden_dims', default=0, nargs='+', type=int, help="dimensions of hidden layers")
    parser.add_argument('--lr', default=0.00005, type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="weight decay")
    parser.add_argument('--l1_reg', default=0.0, type=float, help="l1 regularization")
    parser.add_argument('--patience', default=100, type=int, help="number of iterations we wait before early stopping")

    args = parser.parse_args()

    ## HYPERPARAMS ##
    env_name = args.env
    seed = args.seed
    reward_learning_data_path = args.reward_learning_data_path
    trained_policy_path = args.trained_policy_path
    num_trajs = args.num_trajs
    fully_observable = args.fully_observable
    pure_fully_observable = args.pure_fully_observable
    load_weights = args.load_weights
    discriminator_model_path = args.discriminator_model_path
    hidden_dims = tuple(args.hidden_dims) if args.hidden_dims != 0 else tuple()
    lr = args.lr
    weight_decay = args.weight_decay
    l1_reg = args.l1_reg
    num_epochs = args.num_epochs  # num times through training data
    patience = args.patience

    run(env_name, seed, reward_learning_data_path, trained_policy_path, num_trajs, fully_observable, pure_fully_observable,
        load_weights, discriminator_model_path, hidden_dims, lr, l1_reg, num_epochs, patience)
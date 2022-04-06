import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from sklearn.model_selection import train_test_split


# num_comps specifies the number of pairwise comparisons between trajectories to use in our training set
# pair_delta=1 recovers original (just that pairwise comps can't be the same)
# if all_pairs=True, rather than generating num_comps pairwise comps with pair_delta ranking difference,
# we simply generate all (num_demos choose 2) possible pairs from the dataset.
def create_training_data(demonstrations, num_comps=0, pair_delta=1, all_pairs=False):
    # collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    if all_pairs:
        for ti in range(num_demos):
            for tj in range(ti+1, num_demos):
                traj_i = demonstrations[ti]
                traj_j = demonstrations[tj]

                # In other words, label = (traj_i < traj_j)
                if ti > tj:
                    label = 0  # 0 indicates that traj_i is better than traj_j
                else:
                    label = 1  # 1 indicates that traj_j is better than traj_i

                training_obs.append((traj_i, traj_j))
                training_labels.append(label)

                # We shouldn't need max_traj_length, since all our trajectories our fixed at length 200.
                max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
    else:
        # add full trajs
        for n in range(num_comps):
            ti = 0
            tj = 0
            # only add trajectories that are different returns
            while abs(ti - tj) < pair_delta:
                # pick two random demonstrations
                ti = np.random.randint(num_demos)
                tj = np.random.randint(num_demos)

            traj_i = demonstrations[ti]
            traj_j = demonstrations[tj]

            # In other words, label = (traj_i < traj_j)
            if ti > tj:
                label = 0  # 0 indicates that traj_i is better than traj_j
            else:
                label = 1  # 1 indicates that traj_j is better than traj_i

            training_obs.append((traj_i, traj_j))
            training_labels.append(label)

            # We shouldn't need max_traj_length, since all our trajectories our fixed at length 200.
            max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels


# NOTE:
# Reacher has 11 raw features and 1 privileged feature (distance to target)
class Net(nn.Module):
    def __init__(self, hidden_dims=(128,64), augmented=True, augmented_full=False, num_rawfeatures=11, norm=False):
        super().__init__()

        if augmented_full:
            input_dim = num_rawfeatures + 2
        elif augmented:
            input_dim = num_rawfeatures + 1
        else:
            input_dim = 11

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

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0

        #compute forward pass of reward network (we parallelize across frames so batch size is length of full trajectory)
        x = traj

        # Normalize features
        if self.normalize:
            x = self.layer_norm(x)

        # Propagate through fully-connected layers
        for l in range(self.num_layers - 1):
            x = F.leaky_relu(self.fcs[l](x))
        r = self.fcs[-1](x)

        # Sum across 'batch', which is really the time dimension of the trajectory
        sum_rewards += torch.sum(r)
        return sum_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i = self.cum_return(traj_i)
        cum_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0)


def learn_reward(reward_network, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir, val_obs, val_labels, patience):
    # check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print("device:", device)
    # Note that a sigmoid is implicitly applied in the CrossEntropyLoss
    loss_criterion = nn.CrossEntropyLoss()

    trigger_times = 0
    prev_min_val_loss = 100
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]

            label = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            label = torch.from_numpy(label).to(device)

            # zero out gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            # print("train outputs", outputs.shape)
            # print("train label", label.shape)

            # Calculate loss
            cross_entropy_loss = loss_criterion(outputs, label)
            l1_loss = l1_reg * torch.linalg.vector_norm(torch.cat([param.view(-1) for param in reward_network.parameters()]), 1)
            loss = cross_entropy_loss + l1_loss

            # Backpropagate
            loss.backward()

            # Take one optimizer step
            optimizer.step()

        val_loss = calc_val_loss(reward_network, val_obs, val_labels)
        val_acc = calc_accuracy(reward_network, val_obs, val_labels)
        print("end of epoch {}: val_loss {}, val_acc {}".format(epoch, val_loss, val_acc))

        # Early Stopping
        if val_loss > prev_min_val_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
            if trigger_times >= patience:
                print("Early stopping.")
                return
        else:
            trigger_times = 0
            print('trigger times:', trigger_times)
            print("saving model weights...")
            torch.save(reward_net.state_dict(), checkpoint_dir)
            print("Weights:", reward_net.state_dict())

        prev_min_val_loss = min(prev_min_val_loss, val_loss)
    print("Finished training.")


# Calculates the cross-entropy losses over the entire validation set and returns the MEAN.
def calc_val_loss(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = np.array([training_outputs[i]])
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            label = torch.from_numpy(label).to(device)

            #forward to get logits
            outputs = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)
            # print("val outputs", outputs.shape)
            # print("val label", label.shape)

            loss = loss_criterion(outputs, label)
            losses.append(loss.item())

    return np.mean(losses)


def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs,0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)


def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device)).item()
            rewards_from_obs.append(r)
    return rewards_from_obs


def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


def run(reward_model_path, seed, num_comps=0, num_demos=120, hidden_dims=tuple(), lr=0.00005, weight_decay=0.0, l1_reg=0.0,
        num_epochs=100, patience=100, pair_delta=1, all_pairs=False, augmented=False, augmented_full=False,
        num_rawfeatures=11, normalize_features=False, privileged_reward=False, checkpointed=False, test=False,
        al_data=tuple()):
    if al_data:
        demos = al_data[0]
        demo_rewards = al_data[1]
    else:
        if augmented_full:
            demos = np.load("data/augmented_full/demos.npy")
            demo_rewards = np.load("data/augmented_full/demo_rewards.npy")
            demo_reward_per_timestep = np.load("data/augmented_full/demo_reward_per_timestep.npy")

            raw_features = demos[:, :, 0:num_rawfeatures]  # how many raw features to keep in the observation
            handpicked_features = demos[:, :, 11:13]  # handpicked features are the last 2
            demos = np.concatenate((raw_features, handpicked_features), axis=-1)  # assign the result back to demos
        elif augmented:
            if checkpointed:
                print("Using trajectories from checkpointed policy...")
                demos = np.load("data/checkpointed/augmented/demos.npy")
                demo_rewards = np.load("data/checkpointed/augmented/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/checkpointed/augmented/demo_reward_per_timestep.npy")
            else:
                demos = np.load("data/augmented/demos.npy")
                if privileged_reward:
                    print("Using reward based purely on privileged features...")
                    demo_rewards = np.load("data/augmented/privileged_rewards.npy")
                else:
                    demo_rewards = np.load("data/augmented/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/augmented/demo_reward_per_timestep.npy")

            raw_features = demos[:, :, 0:num_rawfeatures]  # how many raw features to keep in the observation
            handpicked_features = demos[:, :, 11:12]  # handpicked features are the last 1
            demos = np.concatenate((raw_features, handpicked_features), axis=-1)  # assign the result back to demos
        else:
            if checkpointed:
                # TODO: Fill with file paths to checkpointed rollouts
                demos = None
                demos_rewards = None
            else:
                demos = np.load("data/raw/demos.npy")
                demo_rewards = np.load("data/raw/demo_rewards.npy")
                demo_reward_per_timestep = np.load("data/raw/demo_reward_per_timestep.npy")

            if test:
                # TODO: Not implemented
                # Test Data for Vanilla Model
                test_demos = np.load("data/raw_data/test_data/demos.npy")
                test_demo_rewards = np.load("data/raw_data/test_data/demo_rewards.npy")

        print("demos:", demos.shape)
        print("demo_rewards:", demo_rewards.shape)

    # sort the demonstrations according to ground truth reward to simulate ranked demos
    # sorts the demos in order of increasing reward (most negative reward to most positive reward)
    # note that sorted_demos is now a python list, not a np array
    sorted_demos = [x for _, x in sorted(zip(demo_rewards, demos), key=lambda pair: pair[0])]
    sorted_demos = np.array(sorted_demos)
    sorted_demo_rewards = sorted(demo_rewards)
    sorted_demo_rewards = np.array(sorted_demo_rewards)
    print(sorted_demo_rewards)

    if test:
        # Sort test data as well
        sorted_test_demos = [x for _, x in sorted(zip(test_demo_rewards, test_demos), key=lambda pair: pair[0])]
        sorted_test_demos = np.array(sorted_test_demos)
        sorted_test_demo_rewards = sorted(test_demo_rewards)
        sorted_test_demo_rewards = np.array(sorted_test_demo_rewards)


    # Subsample the demos according to num_demos
    # Source: https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
    idx = np.round(np.linspace(0, len(demos) - 1, num_demos)).astype(int)
    sorted_demos = sorted_demos[idx]
    sorted_demo_rewards = sorted_demo_rewards[idx]
    # demo_reward_per_timestep = demo_reward_per_timestep[idx]  # Note: not used.

    train_val_split_seed = 100
    obs, labels = create_training_data(sorted_demos, num_comps, pair_delta, all_pairs)
    if test:
        test_obs, test_labels = create_training_data(sorted_test_demos, all_pairs=True)

    if len(obs) > 1:
        training_obs, val_obs, training_labels, val_labels = train_test_split(obs, labels, test_size=0.10, random_state=train_val_split_seed)
    else:
        print("WARNING: Since there is only one training point, the validation data is the same as the training data.")
        training_obs = val_obs = obs
        training_labels = val_labels = labels

    print("num training_obs", len(training_obs))
    print("num training_labels", len(training_labels))
    print("num val_obs", len(val_obs))
    print("num val_labels", len(val_labels))
    if test:
        print("num test_obs", len(test_obs))
        print("num test_labels", len(test_labels))

    # Now we create a reward network and optimize it using the training data.
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net(hidden_dims=hidden_dims, augmented=augmented, augmented_full=augmented_full, num_rawfeatures=num_rawfeatures, norm=normalize_features)
    reward_net.to(device)
    num_total_params = sum(p.numel() for p in reward_net.parameters())
    num_trainable_params = sum(p.numel() for p in reward_net.parameters() if p.requires_grad)
    print("Total number of parameters:", num_total_params)
    print("Number of trainable paramters:", num_trainable_params)

    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_obs, training_labels, num_iter, l1_reg, reward_model_path, val_obs, val_labels, patience)

    # print out predicted cumulative returns and actual returns
    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in sorted_demos]
    for i, p in enumerate(pred_returns):
        print(i, p, sorted_demo_rewards[i])

    print("train accuracy:", calc_accuracy(reward_net, training_obs, training_labels))
    print("validation accuracy:", calc_accuracy(reward_net, val_obs, val_labels))
    if test:
        print("test accuracy:", calc_accuracy(reward_net, test_obs, test_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--reward_model_path', default='',
                        help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, type=int, help="random seed for experiments")
    parser.add_argument('--num_comps', default=0, type=int, help="number of pairwise comparisons")
    parser.add_argument('--num_demos', default=120, type=int, help="the number of demos to sample pairwise comps from")
    parser.add_argument('--num_epochs', default=100, type=int, help="number of training epochs")
    parser.add_argument('--hidden_dims', default=0, nargs='+', type=int, help="dimensions of hidden layers")
    parser.add_argument('--lr', default=0.00005, type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="weight decay")
    parser.add_argument('--l1_reg', default=0.0, type=float, help="l1 regularization")
    parser.add_argument('--patience', default=100, type=int, help="number of iterations we wait before early stopping")
    parser.add_argument('--pair_delta', default=1, type=int, help="min difference between trajectory rankings in our dataset")
    parser.add_argument('--all_pairs', dest='all_pairs', default=False, action='store_true', help="whether we generate all pairs from the dataset (num_demos choose 2)")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--augmented', dest='augmented', default=False, action='store_true', help="whether data consists of states + linear features pairs rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--augmented_full', dest='augmented_full', default=False, action='store_true', help="whether data consists of states + (distance, action norm) rather that just states")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--num_rawfeatures', default=11, type=int, help="the number of raw features to keep in the augmented space")
    parser.add_argument('--normalize_features', dest='normalize_features', default=False, action='store_true', help="whether to normalize features")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--privileged_reward', dest='privileged_reward', default=False, action='store_true', help="whether to use reward based on privileged features for rankings")  # NOTE: type=bool doesn't work, value is still true.
    parser.add_argument('--checkpointed', dest='checkpointed', default=False, action='store_true', help="checkpointed")

    # TODO: may not have a test dataset for Reacher
    parser.add_argument('--test', dest='test', default=False, action='store_true', help="testing mode for raw observations")
    args = parser.parse_args()

    seed = args.seed

    ## HYPERPARAMS ##
    num_comps = args.num_comps
    num_demos = args.num_demos
    hidden_dims = tuple(args.hidden_dims) if args.hidden_dims != 0 else tuple()
    lr = args.lr
    weight_decay = args.weight_decay
    l1_reg = args.l1_reg
    num_iter = args.num_epochs  # num times through training data
    patience = args.patience
    pair_delta = args.pair_delta
    all_pairs = args.all_pairs
    augmented = args.augmented
    augmented_full = args.augmented_full
    num_rawfeatures = args.num_rawfeatures
    normalize_features = args.normalize_features
    privileged_reward = args.privileged_reward
    checkpointed = args.checkpointed
    test = args.test
    #################

    run(args.reward_model_path, seed, num_comps, num_demos, hidden_dims, lr, weight_decay, l1_reg, num_iter, patience, pair_delta, all_pairs, augmented, augmented_full, num_rawfeatures, normalize_features, privileged_reward, checkpointed, test)

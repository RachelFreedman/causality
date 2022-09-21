import discriminator_kl
import argparse
import numpy as np
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--indvar_str', default='', help="")
    parser.add_argument('--indvar_num', default=0.0, type=float, help="")
    parser.add_argument('--config', default='', help="")
    args = parser.parse_args()

    config = args.config
    fully_observable = False
    pure_fully_observable = False
    if "pure_fully_observable" in config:
        pure_fully_observable = True
    else:
        fully_observable = True

    results = dict()
    results['train_accs'] = []
    results['val_accs'] = []
    results['dkl_pq'] = []
    results['dkl_qp'] = []
    results['symmetric_dkl'] = []
    for seed in [0, 1, 2]:
        env = "Reacher-v2"
        prefix = "/home/jeremy/gym/"

        if fully_observable:
            reward_learning_data_path = prefix + "trex/data/reacher/raw_stateaction/raw_360/demos.npy"
        else:
            reward_learning_data_path = prefix + "trex/data/reacher/pure_fully_observable/demos.npy"

        if config == 'reacher/vanilla/324demos_hdim128-64_stateaction_allpairs_100epochs_10patience_001lr_00001weightdecay':
            if seed == 0:
                trained_policy_path = prefix + "trained_models_reward_learning/" + config + "_seed" + str(
                    seed) + "/sac/ReacherLearnedReward-v0/checkpoint_001123/checkpoint-1123"
            if seed == 1:
                trained_policy_path = prefix + "trained_models_reward_learning/" + config + "_seed" + str(
                    seed) + "/sac/ReacherLearnedReward-v0/checkpoint_002229/checkpoint-2229"
            if seed == 2:
                trained_policy_path = prefix + "trained_models_reward_learning/" + config + "_seed" + str(
                    seed) + "/sac/ReacherLearnedReward-v0/checkpoint_002227/checkpoint-2227"
        elif config == 'reacher/vanilla/120demos_hdim128-64_stateaction_allpairs_100epochs_10patience_001lr_00001weightdecay' and seed == 2:
            trained_policy_path = prefix + "trained_models_reward_learning/" + config + "_seed" + str(seed) + "/sac/ReacherLearnedReward-v0/checkpoint_002230/checkpoint-2230"
        else:
            trained_policy_path = prefix + "trained_models_reward_learning/" + config + "_seed" + str(seed) + "/sac/ReacherLearnedReward-v0/checkpoint_002231/checkpoint-2231"
        discriminator_model_path = prefix + "discriminator_kl_models/" + config + "_seed" + str(seed) + ".params"

        train_acc, val_acc, dkl_pq, dkl_qp = discriminator_kl.run(env, seed, reward_learning_data_path, trained_policy_path,
                                                                  num_trajs=50, fully_observable=fully_observable, pure_fully_observable=pure_fully_observable,
                                                                  load_weights=True, discriminator_model_path=discriminator_model_path,
                                                                  num_epochs=100, hidden_dims=(128, 128, 128), lr=0.01,
                                                                  weight_decay=0.0001, l1_reg=0.0, patience=10)
        results['train_accs'].append(float(train_acc))
        results['val_accs'].append(float(val_acc))
        results['dkl_pq'].append(float(dkl_pq))
        results['dkl_qp'].append(float(dkl_qp))
        results['symmetric_dkl'].append(float(dkl_pq + dkl_qp))

    results['avg_symmetric_dkl'] = float(np.mean(results['symmetric_dkl']))

    outfile = prefix+"discriminator_kl_outputs/"+config+"_results.json"
    with open(outfile, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(results, f, indent=2)


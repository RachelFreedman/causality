#!/bin/bash
#Note that this will only work for our lab Linux machines, since my Mac runs zsh

var1=$1  # This is the variable that will hold our independent variable in our experiments
var2=$2
for seed in 0 1 2; do
  echo Seed $seed

  #Reward-learning
  echo "Reward learning..."
  config="feature_sensitivity/${var1}raw_augmented_linear_2000prefs_60pairdelta_100epochs_10patience_001lr_01l1reg"
  reward_model_path="/home/jeremy/gym/trex/models/${config}_seed${seed}.params"
  reward_output_path="reward_learning_outputs/${config}_seed${seed}.txt"

  cd trex/
  python3 model.py --augmented --num_rawfeatures ${var1} --num_comps 2000 --pair_delta 60 --num_epochs 100 --patience 10 --lr 0.01 --l1_reg 0.1 --seed $seed --reward_model_path $reward_model_path > $reward_output_path

  #RL
  echo "Performing RL..."
  cd ..
  policy_save_dir="./trained_models_reward_learning/${config}_seed${seed}"
  python3 mujoco_gym/learn.py --env "ReacherLearnedReward-v0" --algo sac --seed $seed --train --train-timesteps 1000000 --reward-net-path $reward_model_path --indvar ${var1} --save-dir $policy_save_dir

  #Eval
  echo "Evaluating RL..."
  load_policy_path="${policy_save_dir}/sac/ReacherLearnedReward-v0/checkpoint_002499/checkpoint-2499"
  eval_path="trex/rl/eval/${config}_seed${seed}.txt"
  python3 mujoco_gym/learn.py --env "Reacher-v2" --algo sac --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path $load_policy_path > $eval_path
done




#!/bin/bash
#Note that this will only work for our lab Linux machines, since my Mac runs zsh

var1=$1  # This is the variable that will hold our independent variable in our experiments
var2=$2
for seed in 0 1 2; do
  echo Seed $seed

  #Reward-learning
  echo "Reward learning..."
  config="reacher/vanilla/324demos_hdim128-64_stateaction_allpairs_100epochs_10patience_001lr_00001weightdecay"
  reward_model_path="/home/jeremy/gym/trex/models/${config}_seed${seed}.params"
  reward_output_path="reward_learning_outputs/${config}_seed${seed}.txt"

  cd trex/
#  python3 model.py --env "Reacher-v2" --num_demos 40 --seed $seed --state_action --hidden_dims 256 256 256 --all_pairs --num_epochs 100 --patience 10 --lr 0.0001 --weight_decay 0.0001 --reward_model_path $reward_model_path > $reward_output_path

  #RL
  echo "Performing RL..."
  cd ..
  config="reacher/vanilla/${var1}klpenalty_324demos_hdim128-64_stateaction_allpairs_100epochs_10patience_001lr_00001weightdecay"
  policy_save_dir="./trained_models_reward_learning/${config}_seed${seed}"
  python3 mujoco_gym/learn.py --env "ReacherLearnedReward-v0" --algo sac --seed $seed --train --train-timesteps 1000000 --reward-net-path $reward_model_path --indvar $var1 --save-dir $policy_save_dir --load-policy-path $policy_save_dir --tb

  #Eval
  echo "Evaluating RL..."
  load_policy_path="${policy_save_dir}/sac/ReacherLearnedReward-v0/checkpoint_002231/checkpoint-2231"
  gt_eval_path="trex/rl/eval/${config}_seed${seed}.txt"
  learned_eval_path="trex/rl/eval/${config}_seed${seed}_learnedreward.txt"
  python3 mujoco_gym/learn.py --env "Reacher-v2" --algo sac --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path $load_policy_path > $gt_eval_path
  python3 mujoco_gym/learn.py --env "ReacherLearnedReward-v0" --reward-net-path $reward_model_path --indvar $var1 --algo sac --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path $load_policy_path > $learned_eval_path
done




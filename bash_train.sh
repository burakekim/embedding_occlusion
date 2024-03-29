#!/bin/bash   
# chmod +x bash_train.sh
# TODO add pip install torchgeo to DOCKERFILE, black, isort, tmux ...

#1 is to create test set influence scores

#1-2 should be same performance-wise
#1-2 walltime

#2,3,4,5,6 = modality wise comparison # 2, not 1 cuz 1 has regularization saving in test and this should cause delay. we do not need that for benchmarking

python main_train.py --exp_name 69_all_adamw2_1e_5_heavy_aug_mae --batch_size 16 --max_epochs 50 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization None


# # 1
# python main_train.py --exp_name 61__regularizer_all --batch_size 16 --max_epochs 20 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --test_batch_size 1

# #2
# python main_train.py --exp_name 57__regularizer_all_no_logs --batch_size 32 --max_epochs 100 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization None

# #3
# python main_train.py --exp_name 57__regularizer_s1_no_logs --batch_size 32 --max_epochs 100 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 's1' --use_regularization None

# #4
# python main_train.py --exp_name 57__regularizer_s2_no_logs --batch_size 32 --max_epochs 100 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 's2' --use_regularization None

# #5
# python main_train.py --exp_name 57__regularizer_esa_wc_no_logs --batch_size 32 --max_epochs 100 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'esa_wc' --use_regularization None

# #6
# python main_train.py --exp_name 57__regularizer_viirs_no_logs --batch_size 32 --max_epochs 100 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'viirs' --use_regularization None

######################
#### occlusion strategy

# python main_train.py --exp_name 61__regularizer_zero --batch_size 16 --max_epochs 10 --test_batch_size 1 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "zero"

# python main_train.py --exp_name 61__regularizer_one --batch_size 16 --max_epochs 10 --test_batch_size 1 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "one"

# python main_train.py --exp_name 61__regularizer_random --batch_size 16 --max_epochs 10 --test_batch_size 1 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "random"

# python main_train.py --exp_name 61__regularizer_gaussian --batch_size 16 --max_epochs 10 --test_batch_size 1 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "gaussian"

##############

# # run ckpy with bs of 1 to export correct influence scores
# checkpoint_path = ".logs/57__regularizer/57__regularizer_16_256epoch=51-val_loss=0.02.ckpt"
# python main_train.py --exp_name 57__regularizer --batch_size 16 --max_epochs 100 --test_batch_size 1 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --checkpoint_path "./logs/57__regularizer/57__regularizer_16_256epoch=51-val_loss=0.02.ckpt"



















# # Define the parameters
# run_main="false" # "true" to run the main experiments
# run_modalities="false" # "true" to run the modality benchmark
# run_test_with_checkpoint="false" #  "true" to run tests with checkpoint

# batch_size=16
# max_epochs=500
# crop_size=256
# use_sampler="--use_sampler"
# use_lr_scheduler="--use_lr_scheduler"

# exp_name="55__all"
# return_modality="all"
# checkpoint_path="./logs/${exp_name}/55__all_8_256epoch=14-val_loss=0.02.ckpt"

# ./logs/57__regularizer_all_no_logs/57__regularizer_all_no_logs_32_256epoch=70-val_loss=0.02.ckpt
# ./logs/57__regularizer_s2_no_logs/57__regularizer_s2_no_logs_32_256epoch=62-val_loss=0.02.ckpt
# ./logs/57__regularizer_s1_no_logs/57__regularizer_s1_no_logs_32_256epoch=41-val_loss=0.02.ckpt
# ./logs/57__regularizer_esa_wc_no_logs/57__regularizer_esa_wc_no_logs_32_256epoch=62-val_loss=0.02.ckpt
# ./logs/57__regularizer_viirs_no_logs/57__regularizer_viirs_no_logs_32_256epoch=83-val_loss=0.02.ckpt

# python main_train.py --exp_name 57__regularizer_viirs_no_logs --batch_size 32 --max_epochs 100 \
# --crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'viirs' \
# --use_regularization None --checkpoint_path "./logs/57__regularizer_viirs_no_logs/57__regularizer_viirs_no_logs_32_256epoch=83-val_loss=0.02.ckpt"

####
# Main
####

# if [ "$run_main" = true ]; then
#     # Run the training command without regularization
#     python main_train.py --exp_name ${exp_name}__plain --batch_size ${batch_size} --max_epochs ${max_epochs} --crop_size ${crop_size} ${use_sampler} ${use_lr_scheduler} --return_modality ${return_modality}

#     # Run the training command with regularization
#     python main_train.py --exp_name ${exp_name}__regularizer --batch_size ${batch_size} --max_epochs ${max_epochs} --crop_size ${crop_size} ${use_sampler} ${use_lr_scheduler} --return_modality ${return_modality} --use_regularization
# else
#     echo "Skipping main training commands. Set run_main to true to run them."
# fi

# ####
# # Run test with ckpt
# ####

# if [ "$run_test_with_checkpoint" = "true" ]; then
#     echo "Running Experiment ${exp_name} with checkpoint"
#     python main_train.py --exp_name ${exp_name} --batch_size ${batch_size} --max_epochs ${max_epochs} \
#     --crop_size ${crop_size} --return_modality ${return_modality} --use_sampler \
#     --use_lr_scheduler --checkpoint_path "${checkpoint_path}" 
# else
#     echo 'Skipping test with ckpt. Set "run_test_with_checkpoint" to true to run it.'
# fi

# ####
# # Modality benchmark
# ####

# # Check the variable in a conditional statement
# if [ "$run_modalities" = "true" ]; then
#     echo "Running the loop..."
#     # List of modalities to iterate over
#     modalities=("all" "s1" "s2" "esa_wc" "viirs")

#     # Loop through each modality and run the training command
#     for modality in "${modalities[@]}"
#     do
#         # Customize the experiment name with the current modality as a suffix
#         current_exp_name="${exp_name}_${modality}"
#         echo "Running training for modality: $modality with experiment name: $current_exp_name"
#         python main_train.py --exp_name $current_exp_name --batch_size $batch_size \
#         --max_epochs $max_epochs --crop_size $crop_size \
#         --use_sampler --use_lr_scheduler --return_modality $modality
#     done
# else
#     echo "Skipping modality loop. Set "run_modalities" true to run it."
# fi
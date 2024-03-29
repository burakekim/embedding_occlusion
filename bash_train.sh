#!/bin/bash   

# main 

python main_train.py --exp_name 69_all_adamw2_1e_5_heavy_aug_mae --batch_size 16 --max_epochs 50 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization None

# modality bencmark

python main_train.py --exp_name 61__regularizer_all --batch_size 16 --max_epochs 20 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --test_batch_size 1

python main_train.py --exp_name 57__regularizer_all_no_logs --batch_size 16 --max_epochs 100 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization None

python main_train.py --exp_name 57__regularizer_s1_no_logs --batch_size 16 --max_epochs 100 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 's1' --use_regularization None

python main_train.py --exp_name 57__regularizer_s2_no_logs --batch_size 16 --max_epochs 100 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 's2' --use_regularization None

python main_train.py --exp_name 57__regularizer_esa_wc_no_logs --batch_size 16 --max_epochs 100 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'esa_wc' --use_regularization None

python main_train.py --exp_name 57__regularizer_viirs_no_logs --batch_size 16 --max_epochs 100 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'viirs' --use_regularization None

# occlusion strategy benchmark

python main_train.py --exp_name 61__regularizer_zero --batch_size 16 --max_epochs 10 --test_batch_size 1 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "zero"

python main_train.py --exp_name 61__regularizer_one --batch_size 16 --max_epochs 10 --test_batch_size 1 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "one"

python main_train.py --exp_name 61__regularizer_random --batch_size 16 --max_epochs 10 --test_batch_size 1 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "random"

python main_train.py --exp_name 61__regularizer_gaussian --batch_size 16 --max_epochs 10 --test_batch_size 1 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --occlusion_type "gaussian"

# run ckpy with bs of 1 to export influence scores in expected format

checkpoint_path = ".logs/57__regularizer/57__regularizer_16_256epoch=51-val_loss=0.02.ckpt"
python main_train.py --exp_name 57__regularizer --batch_size 16 --max_epochs 100 --test_batch_size 1 \
--crop_size 256 --use_sampler --use_lr_scheduler --return_modality 'all' --use_regularization "test" --checkpoint_path "./logs/57__regularizer/57__regularizer_16_256epoch=51-val_loss=0.02.ckpt"










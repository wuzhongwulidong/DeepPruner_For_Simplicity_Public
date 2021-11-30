#!/usr/bin/env bash

python train.py \
--model 'DeepPruner' \
--mode test \
--evaluate_only \
--debug_overFit_train '2' \
--learning_rate 0.001 \
--lr_scheduler_type 'MultiStepLR' \
--checkpoint_dir checkpoints/deepPruner \
--dataset_name 'SceneFlow' \
--data_dir data/SceneFlow \
--image_planes 3 \
--batch_size 16 \
--val_batch_size 1 \
--img_height 256 \
--img_width 512 \
--pretrained_netWork author_pretrained_models/DeepPruner-best-sceneflow.tar \
--val_img_height 544 \
--val_img_width 960 \
--max_epoch 500 \
--milestones 20,30,40,50,60,75,85 \
--evaluate_only

#--pretrained_netWork author_pretrained_models/DeepPruner-best-sceneflow.tar \
#--val_img_height 544 \
#--val_img_width 960 \
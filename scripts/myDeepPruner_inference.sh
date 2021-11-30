#!/usr/bin/env bash

# Inference on Scene Flow test set
#CUDA_VISIBLE_DEVICES=0 \
python inference.py \
--mode test \
--pretrained_netWork author_pretrained_models/DeepPruner-best-sceneflow.tar \
--batch_size 1 \
--img_height 544 \
--img_width 960 \
--count_time
exit

#--pretrained_aanet author_pretrained_models/aanet_sceneflow-5aa5a24e.pth \
#--pretrained_aanet pretrained/aanet_sceneflow-5aa5a24e.pth \
#--pretrained_aanet myTrainedModels/aanet_best_36ff7a9e.pth \


# Inference on KITTI 2015 test set for submission
#export CUDA_VISIBLE_DEVICES=0
#python inference.py \
#--mode test \
#--data_dir data/KITTI/kitti_2015/data_scene_flow \
#--dataset_name KITTI2015 \
#--pretrained_aanet myTrainedModels/aanet_best_d1_61f9a784.pth \
#--batch_size 1 \
#--img_height 384 \
#--img_width 1248 \
#--feature_type aanet \
#--feature_pyramid_network \
#--no_intermediate_supervision \
#--output_dir output/kitti15_test
## 2>&1 | tee myDebug_log.txt

# Inference on KITTI 2012 test set for submission
#CUDA_VISIBLE_DEVICES=0
 python inference.py \
--mode test \
--data_dir data/KITTI/kitti_2012/data_stereo_flow \
--dataset_name KITTI2012 \
--pretrained_aanet myTrainedModels/aanet_best_d1_kitti12_442cb8d4.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type aanet \
--feature_pyramid_network \
--no_intermediate_supervision \
--output_dir output/kitti12_test

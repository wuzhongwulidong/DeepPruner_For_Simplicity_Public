# create soft link to SceneFlow dataset
# for 407 server: 在data目录下执行：
ln -s /home1/wzHD/dataSets_for_run/SceneFlow/RGB_images_finallpass SceneFlow
# ln -s /home/wz/dataSets_for_fast_run/SceneFlow/RGB_images_finallpass SceneFlow
mkdir KITTI
ln -s /home1/wzHD/dataSets_for_run/KITTI/Stereo_Evaluation_2012  KITTI/kitti_2012
ln -s /home1/wzHD/dataSets_for_run/KITTI/Stereo_Evaluation_2015  KITTI/kitti_2015
# for 137 server
ln -s /home1/wzHD/wzProjects/pycharmPrj/aanet/data/SceneFlow/ SceneFlow

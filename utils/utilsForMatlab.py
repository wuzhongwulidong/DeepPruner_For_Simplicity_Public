#-- coding:UTF-8 --
import os
# import scipy.io
from models import utils2
import numpy as np
import torch


def getLossRecord(netName="DeepPruner"):
    trainLossKey = '{}_trainLoss'.format(netName)
    trainLoss_dict = {trainLossKey: {'netName': netName, 'epochs': [],
                                     'avgEPE': [],
                                     'avg_d1': [],
                                     'avg_thres1': [],
                                     'avg_thres2': [],
                                     'avg_thres3': [],
                                     'avg_thres10': [],
                                     'avg_thres20': [],
                                     'avg_loss': []}}
    # 记录数据为matlab的mat文件，用于分析和对比
    valLossKey = '{}_valLoss'.format(netName)
    valLoss_dict = {valLossKey: {'netName': netName, 'epochs': [],
                                 'avgEPE': [],
                                 'avg_d1': [],
                                 'avg_thres1': [],
                                 'avg_thres2': [],
                                 'avg_thres3': [],
                                 'avg_thres10': [],
                                 'avg_thres20': []}}
    return trainLoss_dict, trainLossKey, valLoss_dict, valLossKey


def save_loss_for_matlab(trainLoss_dict, valLoss_dict, auxInfo=''):
    destPath = "./myDataAnalysis"
    destName = "AANet_trainLoss_{}".format(auxInfo)
    utils2.check_path(destPath)
    saveDictAsMatlab(os.path.join(destPath, destName), trainLoss_dict)

    destName = "AANet_valLoss_{}".format(auxInfo)
    utils2.check_path(destPath)
    saveDictAsMatlab(os.path.join(destPath, destName), valLoss_dict)


def saveDictAsMatlab(name, dataDict):
    # scipy.io.savemat(name + '.mat', mdict=dataDict)  !!!!!!!!!!!!
    pass


IMAGENET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).view((3, 1, 1))
IMAGENET_STD = torch.Tensor([0.229, 0.224, 0.225]).view((3, 1, 1))


def saveImgErrorAnalysis(index, img_name, dstPath, dstName, left, right, gt_disp, disparity_pyramid, error_hotMap):
    """
    :param index:
    :param img_name:  左图原始图像的名称
    :param path: 保存路径，如'./myDemoData'
    :param dstName: 保存为的文件名(例如：'arrdata.mat')
    :param left: Tensor(cuda)
    :param right: Tensor(cuda)
    :param gt_disp: Tensor(cuda)
    :param disparity_pyramid: [Tensor(cuda),...,Tensor(cuda)] 低分辨率-->高分辨率
    :return:    """
    assert left.size(0) == 1  # [1, C, H ,W]
    # 控制保存图片的数量：SceneFlow的测试集
    if index not in [10, 50, 100, 150, 200, 728, 1456, 2184, 2912, 3640] and index not in range(0, 3470, 100):
        return

    # 注意参与网络训练的图像是做了归一化的。为了使图像开起来更自然，这里去归一化。
    left = (left.detach().clone().squeeze(0).cpu() * IMAGENET_STD + IMAGENET_MEAN) * 255.0  # [1, C, H, W] -> [C, H, W]
    right = (right.detach().clone().squeeze(0).cpu() * IMAGENET_STD + IMAGENET_MEAN) * 255.0

    left = left.permute(1, 2, 0).numpy()  # [C, H, W] -> Matlab的[H, W, C]
    right = right.permute(1, 2, 0).numpy()

    gt_disp = gt_disp.detach().squeeze(0).cpu().numpy()  # [1, H ,W] -> Matlab的[H, W]
    error_hotMap = error_hotMap.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()

    dataDict = {dstName: {'img_name': img_name, 'left_img': left, 'right_img': right, 'gt_disp': gt_disp,
                          'error_hotMap': error_hotMap}}

    for i in range(len(disparity_pyramid)):
        dataDict[dstName]['pred_disp_{}'.format(i)] = disparity_pyramid[i].detach().squeeze(
            0).cpu().numpy()  # [1, H ,W] -> Matlab的[H, W]

    if not os.path.exists(dstPath):
        os.makedirs(dstPath, exist_ok=True)  # explicitly set exist_ok when multi-processing

    fileName = os.path.join(dstPath, dstName)
    saveDictAsMatlab(fileName, dataDict)

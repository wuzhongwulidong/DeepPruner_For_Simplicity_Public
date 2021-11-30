#-- coding:UTF-8 --
import torch
import numpy as np

EPSILON = 1e-8


def epe_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        epe = np.mean(np.abs(d_est - d_gt))
    else:
        epe = torch.mean(torch.abs(d_est - d_gt))

    return epe


def d1_metric(d_est, d_gt, mask, use_np=False):
    """
     D1-all指标：错误视差的比例
     1. 误差绝对值大于3，且误差超过真实值5%，则认为是错误的预测结果。否则认为是正确的预测结果.
     2. 错误预测的像素占总像素个数的比例，即为D1-all指标。
    """
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = (e > 3) & (e / d_gt > 0.05)

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    """
     >npx指标：
     1. 误差绝对值大于n，则认为是错误的预测结果
     2. 错误预测的像素占总像素个数的比例，即为“>npx”指标。
    """
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())  # 注意，这里利用err_mask，巧妙地计算出了误差绝对值大于thres的像素占全部像素的比例。

    return mean

#-- coding:UTF-8 --
from __future__ import print_function

import torch
import torch.utils.data
import numpy as np
import torchvision.utils as vutils


def gen_error_colormap():
    """
    返回的colorMap实际就是一个维度为[10,5]的numpy数组。
    其中，每一行的第1、2个元素为一个范围，第3、4、5个元素为该范围所用的彩色RGB值。
    这样就可以把指定范围内的标量（如视差误差等）换算为彩色RGB值。
    """
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def disp_error_img(D_est_tensor, D_gt_tensor, abs_thres=3., rel_thres=0.05, dilate_radius=1):
    """
    error_image仅用来显示，供人观察。具体地：
    1.仅观察有ground_truth的像素，无ground_truth的像素的误差直接置为0，在图中显示为纯黑色。
    2.依据预测的误差，进行着色。误差从小到大对应：蓝色到红色。误差过大的无效预测（绝对误差超过3，且相对误差超过5%），统一着色为深红色。
    :param D_est_tensor:
    :param D_gt_tensor:
    :param abs_thres:
    :param rel_thres:
    :param dilate_radius:
    :return:
    """
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    mask = D_gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%。
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
    # get colormap
    cols = gen_error_colormap()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)  # 这里之所以构建维度为[B, H, W, 3],是为了方便numpy赋值操作(=cols[i, 2:])的boradcast。
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]

    return torch.from_numpy(np.ascontiguousarray(error_image.transpose([0, 3, 1, 2])))  # [B, H, W, C=3]-->[B, C=3, H, W]


def save_images(writer, mode_tag, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)
    for tag, values in images_dict.items():
        if not isinstance(values, list) and not isinstance(values, tuple):
            values = [values]
        for idx, value in enumerate(values):
            if len(value.shape) == 3:
                value = value[:, np.newaxis, :, :]  # [B,H,W] --> [B,C=1,H,W]
            value = value[:1]  # [B,C,H,W] --> [1,C,H,W],即只取batch中的第一个图，但需保持维度为[B=1, C, H, W]
            value = torch.from_numpy(value)

            image_name = '{}/{}'.format(mode_tag, tag)
            if len(values) > 1:
                image_name = image_name + "_" + str(idx)
            writer.add_image(image_name, vutils.make_grid(value, padding=0, nrow=1, normalize=True, scale_each=True),
                             global_step)


def tensor2numpy(var_dict):
    for key, vars in var_dict.items():
        if isinstance(vars, np.ndarray):
            var_dict[key] = vars
        elif isinstance(vars, torch.Tensor):
            var_dict[key] = vars.data.cpu().numpy()
        else:
            raise NotImplementedError("invalid input type for tensor2numpy")

    return var_dict


def disp_error_hist(pred_disp, gt_disp, max_disp):
    """
    error_image仅用来显示，供人观察。具体地：
    1.仅观察有ground_truth的像素，无ground_truth的像素的误差直接置为0，在图中显示为纯黑色。
    2.依据预测的误差，进行着色。误差从小到大对应：蓝色到红色。误差过大的无效预测（绝对误差超过3，且相对误差超过5%），统一着色为深红色。
    :param D_est_tensor:
    :param D_gt_tensor:
    :param abs_thres:
    :param rel_thres:
    :param dilate_radius:
    :return:
    """
    gt_disp = gt_disp.detach().cpu().numpy()
    pred_disp = pred_disp.detach().cpu().numpy()
    B, H, W = gt_disp.shape
    # valid mask
    mask = np.logical_and(gt_disp > 0, gt_disp < max_disp)

    error_arr = gt_disp - pred_disp
    error_inRange = error_arr[mask]

    return error_inRange


def save_hist(writer, mode_tag, disp_error, global_step, bins=np.arange(-3, 3, 1/200.0) + 0.5 * 1/200):  # bins=np.arange(-5, 5, 1/100)
    disp_error[disp_error < bins[0]] = bins[0]
    disp_error[disp_error > bins[-1]] = bins[-1]

    writer.add_histogram(mode_tag, disp_error, global_step,  bins=bins)  # 这句代码在pytorch0.4.1 torch0.2.0下会报错

#-- coding:UTF-8 --
import os

import torch
import time
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter  # pytorch0.4.1版本还没有引入torch.utils.tensorboard
import torch.nn.functional as F
import os
# from contextlib import nullcontext
# import numpy as np
# import scipy.io
# from utils2 import utils
# from utils.utils import save_checkpoint
# from utils.utils import load_pretrained_net
# from utils.utils import check_path
# sys.path.append('../')

# from utils2 import save_checkpoint, check_path, load_pretrained_net
# from utilsForMatlab import saveImgErrorAnalysis
# from visualization import disp_error_img, save_images, disp_error_hist, save_hist
from models.utils2 import save_checkpoint, check_path, load_pretrained_net
from models.utilsForMatlab import saveImgErrorAnalysis
from models.visualization import disp_error_img, save_images, disp_error_hist, save_hist
from metric import d1_metric, thres_metric


def calc_Loss(pred_disp_pyramid, gt_disp, mask, args, logger, pseudo_mask=None, pseudo_gt_disp=None):
    if args.model == 'DeepPruner':
            loss_weights = {
                'alpha_super_refined': 1.6,
                'alpha_refined': 1.3,
                'alpha_ca': 1.0,
                'alpha_quantile': 1.0,
                'alpha_min_max': 0.7}

            cost_aggregator_scale = 4
            # forces min_disparity to be equal or slightly lower than the true disparity
            quantile_mask1 = ((gt_disp[mask] - pred_disp_pyramid[-1][mask]) < 0).float()
            quantile_loss1 = (gt_disp[mask] - pred_disp_pyramid[-1][mask]) * (0.05 - quantile_mask1)
            quantile_min_disparity_loss = quantile_loss1.mean()

            # forces max_disparity to be equal or slightly larger than the true disparity
            quantile_mask2 = ((gt_disp[mask] - pred_disp_pyramid[-2][mask]) < 0).float()
            quantile_loss2 = (gt_disp[mask] - pred_disp_pyramid[-2][mask]) * (0.95 - quantile_mask2)
            quantile_max_disparity_loss = quantile_loss2.mean()

            min_disparity_loss = F.smooth_l1_loss(pred_disp_pyramid[-1][mask], gt_disp[mask], size_average=True)
            max_disparity_loss = F.smooth_l1_loss(pred_disp_pyramid[-2][mask], gt_disp[mask], size_average=True)
            ca_depth_loss = F.smooth_l1_loss(pred_disp_pyramid[-3][mask], gt_disp[mask], size_average=True)
            refined_depth_loss = F.smooth_l1_loss(pred_disp_pyramid[-4][mask], gt_disp[mask], size_average=True)

            if cost_aggregator_scale == 8:
                refined_depth_loss_1 = F.smooth_l1_loss(pred_disp_pyramid[-5][mask], gt_disp[mask], size_average=True)
                loss = (loss_weights.alpha_super_refined * refined_depth_loss_1)
                output_disparity = pred_disp_pyramid[-5]
                # logging.info('refined_depth_loss_1: %.6f', refined_depth_loss_1)
            else:
                loss = 0
                output_disparity = pred_disp_pyramid[-4]

            loss += (loss_weights.alpha_refined * refined_depth_loss) + \
                    (loss_weights.alpha_ca * ca_depth_loss) + \
                    (loss_weights.alpha_quantile * (quantile_max_disparity_loss + quantile_min_disparity_loss)) + \
                    (loss_weights.alpha_min_max * (min_disparity_loss + max_disparity_loss))

            return loss, [output_disparity], loss, None, None, None
    else:
        logger.info('=>calc_Loss: UnSupported NetWork: {}!!!!!!!!!!!!!'.format(args.model))
        raise NotImplementedError


def getPredictedDisparity(result, gt_disp, mask, args, logger):
    if args.model == 'aanet':
        pred_disp_pyramid = result[-1]
        return pred_disp_pyramid
    elif args.model == 'DeepPruner':
        if args.mode == 'test':  # myDeepPruner_evaluate.sh中mode=test，则对应地也需要修改models/config.py中的mode=evaluation。
            pred_disp_pyramid = result  # 要：当models/config.py中的mode=evaluation时，result中仅有最终预测的视差
        else:
            cost_aggregator_scale = 4  # 重要：必须与models/config.py中的cost_aggregator_scale数值保持一致
            if cost_aggregator_scale == 8:
                pred_disp_pyramid = result[-5]
            else:
                pred_disp_pyramid = result[-4]

        return [pred_disp_pyramid]
    else:
        logger.info('=>calc_Loss: UnSupported NetWork: {}!!!!!!!!!!!!!'.format(args.model))
        raise NotImplementedError


def getDispMask(gt_disp, sample, args, device):
    mask = (gt_disp > 0) & (gt_disp < args.max_disp)  # KITTI数据集约定：视差为0，表示无效视差。
    if args.load_pseudo_gt:
        pseudo_gt_disp = sample['pseudo_disp'].to(device)
        pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < args.max_disp) & (~mask)  # inverse mask # 需要修补的像素位置的mask
    else:
        pseudo_gt_disp = None
        pseudo_mask = None

    return mask, pseudo_mask, pseudo_gt_disp


def load_pretrained_net_accordingly(netWork, args, logger):
    logger.info('=> loading pretrained model: {}'.format(args.pretrained_netWork))
    load_pretrained_net(netWork, args.pretrained_netWork, no_strict=True)


class mySummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', doTensorBoardSummary=True):
        super(mySummaryWriter, self).__init__(log_dir, comment, purge_step, max_queue,
                 flush_secs, filename_suffix)
        self.doTensorBoardSummary = doTensorBoardSummary

    # 重写父类方法
    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.doTensorBoardSummary:
            super(mySummaryWriter, self).add_scalar(tag, scalar_value, global_step, walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        if self.doTensorBoardSummary:
            super(mySummaryWriter, self).add_image(tag, img_tensor, global_step, walltime, dataformats)


class netWorkModel(object):
    def __init__(self, args, logger, optimizer, netWork, device, start_iter=0, start_epoch=0,
                 best_epe=None, best_epoch=None):
        self.args = args
        self.logger = logger
        self.optimizer = optimizer
        self.netWork = netWork
        self.device = device
        self.num_iter = start_iter
        self.epoch = start_epoch

        self.best_epe = 999. if best_epe is None else best_epe
        self.best_epoch = -1 if best_epoch is None else best_epoch

        self.best_d1 = 999.
        self.best_epoch_d1 = -1

    def val_only_load_pretrained_net_from_file(self):
        load_pretrained_net_accordingly(self.netWork, self.args, self.logger)

    def validate(self, val_loader, local_master=None, valLossDict=None, valLossKey=None):
        args = self.args
        logger = self.logger
        logger.info('=> Start validation...')

        self.netWork.eval()

        num_samples = len(val_loader)
        logger.info('=> %d samples found in the validation set' % num_samples)

        val_epe = 0
        val_d1 = 0
        val_thres1 = 0
        val_thres2 = 0
        val_thres3 = 0
        val_thres10 = 0
        val_thres20 = 0

        val_count = 0
        val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')

        num_imgs = 0
        valid_samples = 0
        # 遍历验证样本或测试样本
        for i, sample in enumerate(val_loader):
            if (i+1) % 100 == 0:
                logger.info('=> Validating %d/%d' % (i, num_samples))

            left = sample['left'].to(self.device)  # [B, 3, H, W]
            right = sample['right'].to(self.device)
            gt_disp = sample['disp'].to(self.device)  # [B, H, W]

            # mask = (gt_disp > 0) & (gt_disp < args.max_disp)  # !!!!
            mask = (gt_disp > 0) & (gt_disp < args.max_disp)   # !!!!
            # mask = gt_disp >= 0  # !!!!
            # mask = gt_disp > -1  # DeepPruner的评测方式？？!!!!
            # mask = gt_disp < args.max_disp  # DeepPruner的评测方式？？!!!!
            # mask = (gt_disp > 0) & (gt_disp < 230)

            if not mask.any():
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>not mask.any(): i = {} ！！！'.format(i))
                continue

            valid_samples += 1
            num_imgs += gt_disp.size(0)

            with torch.no_grad():
                result = self.netWork(left, right)  # [B, H, W]
                # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>i={}'.format(i))

                disparity_pyramid = getPredictedDisparity(result, gt_disp, mask, args, logger)  # 网络输出整理
                pred_disp = disparity_pyramid[-1]  # 取出最高分辨率的预测视差图

                # padMargin = 5
                # pred_disp = pred_disp[:, padMargin:, :]
                # left = left[:, :, padMargin:, :]
                # right = right[:, :, padMargin:, :]
                # gt_disp = gt_disp[:, padMargin:, :]
                # mask = mask[:, padMargin:, :]

            if pred_disp.size(-1) < gt_disp.size(-1):
                print('------------------------- pred_disp.size(-1) < gt_disp.size(-1)')
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
                                          mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            # epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='elementwise_mean')  # pytorch0.4.1
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
            thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
            thres10 = thres_metric(pred_disp, gt_disp, mask, 10.0)
            thres20 = thres_metric(pred_disp, gt_disp, mask, 20.0)

            val_epe += epe.item()
            val_d1 += d1.item()
            val_thres1 += thres1.item()
            val_thres2 += thres2.item()
            val_thres3 += thres3.item()
            val_thres10 += thres10.item()
            val_thres20 += thres20.item()

        # 遍历验证样本或测试样本完成

        logger.info('=> Validation done!')

        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples
        mean_thres1 = val_thres1 / valid_samples
        mean_thres2 = val_thres2 / valid_samples
        mean_thres3 = val_thres3 / valid_samples
        mean_thres10 = val_thres10 / valid_samples
        mean_thres20 = val_thres20 / valid_samples

        # Save validation results
        with open(val_file, 'a') as f:
            f.write('epoch: %03d\t' % self.epoch)
            f.write('epe: %.3f\t' % mean_epe)
            f.write('d1: %.4f\t' % mean_d1)
            f.write('thres1: %.4f\t' % mean_thres1)
            f.write('thres2: %.4f\t' % mean_thres2)
            f.write('thres3: %.4f\t' % mean_thres3)
            f.write('thres10: %.4f\t' % mean_thres10)
            f.write('thres20: %.4f\n' % mean_thres20)
            f.write('dataset_name= %s\t mode=%s\n' % (args.dataset_name, args.mode))

        logger.info('=> Mean validation epe of epoch %d: %.3f' % (self.epoch, mean_epe))



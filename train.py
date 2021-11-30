#-- coding:UTF-8 --
import torch
from torch.cuda import synchronize
from torch import distributed

import argparse
import numpy as np
import os

from dataloader.dataloader import getDataLoader
from models import network_model
from utils import utils
from utils.utilsForMatlab import getLossRecord, save_loss_for_matlab

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='DeepPruner Arguments')
parser.add_argument('--model', default='DeepPruner', help='select a model structure')
parser.add_argument('--mode', default='val', help='select a model structure')
parser.add_argument('--checkpoint_dir', required=True, help='save path')
parser.add_argument('--pretrained_netWork', default=None, type=str, help='Pretrained network')
# parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
parser.add_argument('--accumulation_steps', default=1, type=int, help='Batch size for training')

# parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility')  # DeepPruner
# # Learning rate
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=64, type=int, help='Batch size for validation')
parser.add_argument('--freeze_bn', action='store_true', help='Switch BN to eval mode to fix running statistics')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Decay gamma')
parser.add_argument('--lr_scheduler_type', default='MultiStepLR', help='Type of learning rate scheduler')
parser.add_argument('--milestones', default=None, type=str, help='Milestones for MultiStepLR')
parser.add_argument('--max_epoch', default=64, type=int, help='Maximum epoch number for training')
# # dataSet Config
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')
# # 用于选择数据集自己选择：0: debug; 1: overFit; 2: Train
parser.add_argument('--debug_overFit_train', default=1, type=int, help='For code debug only!')
parser.add_argument('--data_dir', default='data/SceneFlow', type=str, help='Training dataset')
parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')
# # image size
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--image_planes', default=3, type=int, help='原始图像的通道数')
parser.add_argument('--img_height', default=288, type=int, help='Image height for training')
parser.add_argument('--img_width', default=512, type=int, help='Image width for training')
# # # For KITTI, using 384x1248 for validation
parser.add_argument('--val_img_height', default=576, type=int, help='Image height for validation')
parser.add_argument('--val_img_width', default=960, type=int, help='Image width for validation')
# # parser.add_argument('--padMargin', required=True, type=int, help='Image padMargin')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
#
# # Log
parser.add_argument('--doTensorBoardSummary', action='store_true', help='Whether do TensorBoard Summary?')
parser.add_argument('--print_freq', default=50, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
parser.add_argument('--no_build_summary', action='store_true', help='Dont save sammary when training to save space')
parser.add_argument('--save_ckpt_freq', default=5, type=int, help='Save checkpoint frequency (epochs)')
#
parser.add_argument('--evaluate_only', action='store_true', help='Evaluate pretrained models')
parser.add_argument('--no_validate', action='store_true', help='No validation')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')
parser.add_argument('--highest_loss_only', action='store_true', help='Only use loss on highest scale for finetuning')
parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')

#  尝试分布式训练:
parser.add_argument("--local_rank", type=int)  # 必须有这一句，但是local_rank是torch.distributed.launch自动分配和传入的。
parser.add_argument("--distributed", action='store_true', help="use DistributedDataParallel")

args = parser.parse_args()
utils.check_path(args.checkpoint_dir)
logger = utils.get_logger(os.path.join(args.checkpoint_dir, "trainLog.txt"))

# 0: debug;
# 1: overFit;
# 1x: 子集训练;
# 2: Train使用全量的训练集
# 调整打印频率
if args.debug_overFit_train in [0, 2]:
    args.print_freq = 50
    args.summary_freq = 100
elif args.debug_overFit_train in [1]:
    args.print_freq = 10
    args.summary_freq = 50

if args.dataset_name == 'KITTI2015':
    args.print_freq = 20

if args.distributed:
    #  尝试分布式训练
    # local_rank = torch.distributed.get_rank()
    # local_rank表示本台机器上的进程序号,是由torch.distributed.launch自动分配和传入的。
    local_rank = args.local_rank
    # 根据local_rank来设定当前使用哪块GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # 初始化DDP，使用默认backend(nccl)就行
    torch.distributed.init_process_group(backend="nccl")
    print("args.local_rank={}".format(args.local_rank))
else:
    device = torch.device("cuda")

# 尝试分布式训练
local_master = True if not args.distributed else args.local_rank == 0
utils.save_args(args) if local_master else None

# 打印所用的参数
if local_master:
    logger.info('[Info] used parameters: {}'.format(vars(args)))

torch.backends.cudnn.benchmark = True  # https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317

utils.check_path(args.checkpoint_dir)
utils.save_args(args) if local_master else None

filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
utils.save_command(args.checkpoint_dir, filename) if local_master else None


def loadNetWorks(args):
    if args.model == 'DeepPruner':
        from models.deeppruner import loadNetWorks
        netWork = loadNetWorks(args).cuda()
    else:
        logger.info('=>loadNetWorks: UnSupported NetWork: {} !!!!!!!!!!!!!'.format(args.model))
        raise NotImplementedError
    return netWork


def initOptimizer(args, netWork):
    if args.model in ['DeepPruner']:
        optimizer = torch.optim.Adam(netWork.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    else:
        logger.info('=>Optimizer: UnSupported model: {} !!!!!!!!!!!!!'.format(args.model))
        raise NotImplementedError

    return optimizer


def initLR_Scheduler(optimizer, start_epoch, args, max_epoch=None, steps_per_epoch=None):
    if args.lr_scheduler_type is not None:
        last_epoch = start_epoch if args.resume else start_epoch - 1
        if args.lr_scheduler_type == 'MultiStepLR':
            milestones = [int(step) for step in args.milestones.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones,
                                                                gamma=args.lr_decay_gamma,
                                                                last_epoch=last_epoch)  # 最后这个last_epoch参数很重要：如果是resume的话，则会自动调整学习率适去应last_epoch。
            logger.info('=>lr_scheduler.get_lr():{}'.format(lr_scheduler.state_dict()))
        else:
            logger.info('=>Error: args.lr_scheduler_type cannot be empty!')
            raise NotImplementedError
    else:
        logger.info('=>Error: args.lr_scheduler_type cannot be None!')
        raise NotImplementedError
    return lr_scheduler


def loadModels(args, logger, optimizer, netWork, device, start_iter, start_epoch, best_epe, best_epoch):

    train_model = network_model.netWorkModel(args, logger, optimizer, netWork, device, start_iter, start_epoch,
                                             best_epe=best_epe, best_epoch=best_epoch)
    return train_model


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, val_loader = getDataLoader(args, logger)
    # 根据字符串args.model加载对应的模型
    netWork = loadNetWorks(args)

    if local_master:
        structure_of_net = os.path.join(args.checkpoint_dir, 'structure_of_net.txt')
        with open(structure_of_net, 'w') as f:
            f.write('%s' % netWork)

    if args.pretrained_netWork is not None:
        logger.info('=> Loading pretrained weights: %s' % args.pretrained_netWork)
        # Enable training from a partially pretrained model
        utils.load_pretrained_net(netWork, args.pretrained_netWork, no_strict=(not args.strict))

    netWork.to(device)

    logger.info('=> Use %d GPUs' % torch.cuda.device_count()) if local_master else None

    if args.distributed:
        #  尝试分布式训练
        netWork = torch.nn.SyncBatchNorm.convert_sync_batchnorm(netWork)
        netWork = torch.nn.parallel.DistributedDataParallel(netWork, device_ids=[local_rank],
                                                          output_device=local_rank)
        synchronize()

    # Save parameters
    num_params = utils.count_parameters(netWork)
    logger.info('=> Number of trainable parameters: %d' % num_params)
    save_name = '%d_parameters' % num_params
    open(os.path.join(args.checkpoint_dir, save_name), 'a').close() if local_master else None # 这是个空文件，只是通过其文件名称指示模型有多少个需要训练的参数

    # Optimizer
    optimizer = initOptimizer(args, netWork)

    start_epoch = 0
    start_iter = 0
    best_epe = None
    best_epoch = None

    # model.Model(object)对netWork做了进一步封装。
    train_model = loadModels(args, logger, optimizer, netWork, device, start_iter, start_epoch,
                                          best_epe=best_epe, best_epoch=best_epoch)

    trainLoss_dict, trainLossKey, valLoss_dict, valLossKey = getLossRecord(netName="DeepPruner")

    if args.evaluate_only:
        assert args.val_batch_size == 1
        # 只做evaluate，则需要从文件加载训练好的模型。否则，直接使用本train_model类中保存的(尚未完成全部的Epoach训练的) self.netWork即可。
        train_model.val_only_load_pretrained_net_from_file()
        train_model.validate(val_loader, local_master, valLoss_dict, valLossKey)  # test模式。应该设置--evaluate_only，且--mode为“test”。


if __name__ == '__main__':
    main()

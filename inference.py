#-- coding:UTF-8 --
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# import skimage.io
import argparse
import numpy as np
import time
import os

import models
import dataloader
from dataloader import transforms
from models import network_model
from models.deeppruner import DeepPruner
from utils import utils
from utils.file_io import write_pfm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser(description='DeepPruner Arguments')
parser.add_argument('--mode', default='test', type=str,
                    help='Validation mode on small subset or test mode on full test data')
parser.add_argument('--pretrained_netWork', default=None, type=str, help='Pretrained network')
parser.add_argument('--seed', default=1, type=int, help='Random seed for reproducibility')  # DeepPruner
parser.add_argument('--batch_size', required=True, type=int, help='Batch size for training')
parser.add_argument('--img_height', required=True, type=int, help='Image height for training')
parser.add_argument('--img_width', required=True, type=int, help='Image width for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading')
parser.add_argument('--output_dir', default='output', type=str, help='Directory to save inference results')

# choose data set
# 用于选择数据集：0: debug; 1: overFit; 2: Train
parser.add_argument('--debug_overFit_train', default=2, type=int, help='use author original file name list Dir')
parser.add_argument('--data_dir', default='data/SceneFlow',
                    type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')
parser.add_argument('--save_type', default='png', choices=['pfm', 'png', 'npy'], help='Save file type')
parser.add_argument('--visualize', action='store_true', help='Visualize disparity map')
# Log
parser.add_argument('--count_time', action='store_true', help='Inference on a subset for time counting only')
parser.add_argument('--num_images', default=100, type=int, help='Number of images for inference')

args = parser.parse_args()

model_name = os.path.basename(args.pretrained_netWork)[:-4]
model_dir = os.path.basename(os.path.dirname(args.pretrained_netWork))
args.output_dir = os.path.join(args.output_dir, model_dir + '-' + model_name)

utils.check_path(args.output_dir)
utils.save_command(args.output_dir)


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    test_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                         debug_overFit_train=args.debug_overFit_train,
                                         dataset_name=args.dataset_name,
                                         mode=args.mode,
                                         save_filename=True,
                                         transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    net = DeepPruner().to(device)

    # print(DeepPruner)

    if os.path.exists(args.pretrained_netWork):
        print('=> Loading pretrained DeepPruner:', args.pretrained_netWork)
        utils.load_pretrained_net(net, args.pretrained_netWork, no_strict=True)
    else:
        print('=> Using random initialization')

    # Save parameters
    num_params = utils.count_parameters(net)
    print('=> Number of trainable parameters: %d' % num_params)

    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        net = torch.nn.DataParallel(net)

    # Inference
    net.eval()

    inference_time = 0
    num_imgs = 0

    num_samples = len(test_loader)
    print('=> %d samples found in the test set' % num_samples)

    for i, sample in enumerate(test_loader):
        if args.count_time and i >= args.num_images:  # testing time only, on args.num_images images.
            break

        if i % 20 == 0:
            print('=> Inferencing %d/%d' % (i, num_samples))

        left = sample['left'].to(device)  # [B, 3, H, W]
        right = sample['right'].to(device)

        # Pad
        ori_height, ori_width = left.size()[2:]
        if ori_height < args.img_height or ori_width < args.img_width:
            top_pad = args.img_height - ori_height
            right_pad = args.img_width - ori_width

            # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
            # left = F.pad(left, (0, right_pad, top_pad, 0))
            # right = F.pad(right, (0, right_pad, top_pad, 0))
            # modified by wuzhong
            left = F.pad(left, (0, right_pad, top_pad, 0), mode="reflect")
            right = F.pad(right, (0, right_pad, top_pad, 0), mode="reflect")

        # Warpup
        if i == 0 and args.count_time:
            with torch.no_grad():
                for _ in range(10):
                    net(left, right)

        num_imgs += left.size(0)

        with torch.no_grad():
            # time_start = time.perf_counter()
            # pred_disp = net(left, right)  # [B, H, W]
            # inference_time += time.perf_counter() - time_start

            time_start = datetime.datetime.now()
            pred_disp = net(left, right)  # [B, H, W]
            inference_time += (datetime.datetime.now() - time_start).total_seconds()
            # print("time per image ={}".format((datetime.datetime.now() - time_start).total_seconds()))

    print('=> Mean inference time for %d images: %.3fs' % (num_imgs, inference_time / num_imgs))


if __name__ == '__main__':
    main()

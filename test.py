from model import Predictor
from dataloader import MovingMNIST, KTH
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lpips
import argparse
import numpy as np
import os
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movingmnist',
                    help='testing dataset (movingmnist or kth)')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--make_frame', type=bool, default=True,
                    help='whether to generate predicted frames')
parser.add_argument('--evaluate', type=bool, default=False,
                    help='whether to evaluate performance')
parser.add_argument('--test_data_dir', type=str, default='enter_the_path',
                    help='directory of test set')
parser.add_argument('--test_result_dir', type=str, default='./test_results',
                    help='directory for saving predicted frames')
parser.add_argument('--checkpoint_load_file', type=str, default='enter_the_path',
                    help='file path for loading checkpoint')

parser.add_argument('--img_size', type=int, default=64,
                    help='height and width of video frame')
parser.add_argument('--img_channel', type=int, default=1,
                    help='channel of video frame')
parser.add_argument('--memory_size', type=int, default=100,
                    help='memory slot size')
parser.add_argument('--short_len', type=int, default=10,
                    help='number of input short-term frames')
parser.add_argument('--out_len', type=int, default=30,
                    help='number of output predicted frames')

parser.add_argument('--batch_size', type=int, default=8,
                    help='mini-batch size')
args = parser.parse_args()

if __name__ == '__main__':
    if not os.path.isdir(args.test_result_dir):
        os.makedirs(args.test_result_dir)

    # define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_model = Predictor(args).to(device)
    pred_model = nn.DataParallel(pred_model)

    # load checkpoint
    pred_model.load_state_dict(torch.load(args.checkpoint_load_file))
    print('Checkpoint is loaded from ' + args.checkpoint_load_file)

    # prepare dataloader for selected dataset
    if args.dataset == 'movingmnist':
        test_dataset = MovingMNIST(args.test_data_dir, seq_len=args.short_len+args.out_len, train=False)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    elif args.dataset == 'kth':
        test_dataset = KTH(args.test_data_dir, seq_len=args.short_len+args.out_len, train=False)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    clips = testloader.sampler.data_source.clips
    lpips_dist = lpips.LPIPS(net = 'alex').to(device)
    valid_mse, valid_psnr, valid_ssim, valid_lpips = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    print('Start testing...')
    pred_model.eval()
    with torch.no_grad():
        for test_i, test_data in enumerate(testloader):
            # define data indexes
            short_data_start, short_data_end = 0, args.short_len
            out_gt_start, out_gt_end = short_data_end, short_data_end+args.out_len

            # obtain input data and output gt
            test_data = torch.stack(test_data).to(device)
            test_data = test_data.transpose(dim0=0, dim1=1)
            short_data = test_data[:, short_data_start:short_data_end, :, :, :]
            out_gt = test_data[:, out_gt_start:out_gt_end, :, :, :]

            # frame prediction
            out_pred = pred_model(short_data, None, args.out_len, phase=2)
            out_pred = torch.clamp(out_pred, min = 0, max = 1)

            # calculate evaluation metrics
            batch_size_current = test_data.shape[0]
            if args.evaluate:
                mse, psnr, ssim, lpips = calculate_metrics(out_pred, out_gt, lpips_dist, args)
                valid_mse.update(np.mean(mse), batch_size_current)
                valid_psnr.update(np.mean(psnr), batch_size_current)
                valid_ssim.update(np.mean(ssim), batch_size_current)
                valid_lpips.update(np.mean(lpips), batch_size_current)

            # generate predicted frames
            if args.make_frame:
                for batch_i in range(batch_size_current):
                    video_i, frame_start = clips[test_i*args.batch_size+batch_i]
                    if not os.path.isdir(args.test_result_dir + '/video_'+ str(video_i)+'_' + str(frame_start)):
                        os.makedirs(args.test_result_dir + '/video_'+ str(video_i)+'_' + str(frame_start))
                    for frame_i in range(args.short_len):
                        cv2.imwrite(args.test_result_dir + '/video_'+ str(video_i)+'_' + str(frame_start)+ '/input_'
                                    +str(frame_i).zfill(5) + '.jpg', short_data[batch_i,frame_i,0,:,:].cpu().numpy()*255)
                    for frame_i in range(args.out_len):
                        cv2.imwrite(args.test_result_dir+'/video_'+str(video_i)+'_'+str(frame_start)+'/pred_'+
                                    str(frame_i+args.short_len).zfill(5)+'.jpg', out_pred[batch_i,frame_i,0,:,:].cpu().numpy()*255)

    if args.evaluate:
        print('************** test_output_length [{}] **************'
                  .format(args.out_len))
        print('mse: {:.3f}, psnr: {:.3f}, ssim: {:.3f}, lpips: {:.3f}'
                  .format(valid_mse.avg, valid_psnr.avg, valid_ssim.avg, valid_lpips.avg))
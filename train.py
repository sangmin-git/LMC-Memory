from model import Predictor
from dataloader import MovingMNIST, KTH
from utils import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import lpips
import argparse
import numpy as np
import time
import os


seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movingmnist',
                    help='training dataset (movingmnist or kth)')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--train_data_dir', type=str, default='enter_the_path',
                    help='directory of training set')
parser.add_argument('--valid_data_dir', type=str, default='enter_the_path',
                    help='directory of validation set')
parser.add_argument('--checkpoint_load', type=bool, default=False,
                    help='whether to load checkpoint')
parser.add_argument('--checkpoint_load_file', type=str, default='enter_the_path',
                    help='file path for loading checkpoint')
parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints',
                    help='directory for saving checkpoints')

parser.add_argument('--img_size', type=int, default=64,
                    help='height and width of video frame')
parser.add_argument('--img_channel', type=int, default=1,
                    help='channel of video frame')
parser.add_argument('--memory_size', type=int, default=100,
                    help='memory slot size')
parser.add_argument('--short_len', type=int, default=10,
                    help='number of input short-term frames')
parser.add_argument('--long_len', type=int, default=30,
                    help='number of input long-term frames')
parser.add_argument('--out_len', type=int, default=30,
                    help='number of output predicted frames')

parser.add_argument('--batch_size', type=int, default=128,
                    help='mini-batch size')
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--iterations', type=int, default=300000,
                    help='number of total iterations')
parser.add_argument('--iterations_warmup', type=int, default=5000,
                    help='number of iterations for warming up model')
parser.add_argument('--print_freq', type=int, default=1000,
                    help='frequency of printing logs')
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.isdir(args.checkpoint_save_dir):
        os.makedirs(args.checkpoint_save_dir)

    # define the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_model = Predictor(args).to(device)
    pred_model = nn.DataParallel(pred_model)

    # optionally load checkpoint
    if args.checkpoint_load:
        pred_model.load_state_dict(torch.load(args.checkpoint_load_file))
        print('Checkpoint is loaded from ' + args.checkpoint_load_file)

    # prepare dataloader for selected dataset
    if args.dataset == 'movingmnist':
        train_dataset = MovingMNIST(args.train_data_dir, seq_len=args.short_len+args.out_len, train=True)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        valid_dataset = MovingMNIST(args.valid_data_dir, seq_len=args.short_len+args.out_len, train=False)
        validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    elif args.dataset == 'kth':
        train_dataset = KTH(args.train_data_dir, seq_len=args.short_len+args.out_len, train=True)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        valid_dataset = KTH(args.valid_data_dir, seq_len=args.short_len+args.out_len, train=False)
        validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    # define optimizer and loss function
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr)
    l1_loss, l2_loss = nn.L1Loss().to(device), nn.MSELoss().to(device)
    lpips_dist = lpips.LPIPS(net = 'alex').to(device)

    mse_min, psnr_max, ssim_max, lpips_min = 99999, 0, 0, 99999
    train_loss = AverageMeter()
    valid_mse, valid_psnr, valid_ssim, valid_lpips = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    print('Start training...')
    start_time = time.time()
    data_iterator = iter(trainloader)
    for train_i in range(args.iterations):
        try:
            train_data = next(data_iterator)
        except:
            data_iterator = iter(trainloader)
            train_data = next(data_iterator)

        # define data indexes
        short_start, short_end = 0, args.short_len
        long_start = np.random.randint(0, args.short_len+args.out_len-args.long_len+1)
        long_end = long_start+args.long_len
        out_gt_start, out_gt_end = short_end, short_end+args.out_len

        # obtain input data and output gt
        train_data = torch.stack(train_data).to(device)
        train_data = train_data.transpose(dim0=0, dim1=1) # make (N, T, C, H, W)
        short_data = train_data[:, short_start:short_end, :, :, :]
        long_data = train_data[:, long_start:long_end, :, :, :]
        out_gt = train_data[:, out_gt_start:out_gt_end, :, :, :]

        # predict only 10 frames in the first few iterations to warm up the model
        if (not args.checkpoint_load) and (train_i < args.iterations_warmup):
            train_out_len = 10
            long_data = train_data[:, short_start:out_gt_start+train_out_len, :, :, :]
            out_gt = train_data[:, out_gt_start:out_gt_start+train_out_len, :, :, :]
        else:
            train_out_len = args.out_len

        pred_model.train()

        # training phase 1 with long-term sequence
        pred_model.module.memory.memory_w.requires_grad = True # train memory weights
        out_pred = pred_model(short_data, long_data, train_out_len, phase=1)
        loss_p1 = l1_loss(out_pred, out_gt) + l2_loss(out_pred, out_gt)
        optimizer.zero_grad()
        loss_p1.backward()
        optimizer.step()
        # training phase 2 without long-term sequence
        pred_model.module.memory.memory_w.requires_grad = False # do not train memory weights
        out_pred = pred_model(short_data, None, train_out_len, phase=2)
        loss_p2 = l1_loss(out_pred, out_gt) + l2_loss(out_pred, out_gt)
        optimizer.zero_grad()
        loss_p2.backward()
        optimizer.step()

        train_loss.update(float(loss_p1) +float(loss_p2))

        if (train_i+1) % args.print_freq == 0:
            torch.save(pred_model.state_dict(), args.checkpoint_save_dir+'/trained_file_'+str(train_i+1).zfill(6)+'.pt')

            # validation phase
            pred_model.eval()
            with torch.no_grad():
                for valid_data in validloader:
                    # define data indexes
                    short_start, short_end = 0, args.short_len
                    out_gt_start, out_gt_end = short_end, short_end+args.out_len

                    # obtain input data and output gt
                    valid_data = torch.stack(valid_data).to(device)
                    valid_data = valid_data.transpose(dim0=0, dim1=1) # make (N, T, C, H, W)
                    short_data = valid_data[:, short_start:short_end, :, :, :]
                    out_gt = valid_data[:, out_gt_start:out_gt_end, :, :, :]

                    # frame prediction and calculate evaluation metrics
                    out_pred = pred_model(short_data, None, args.out_len, phase=2)
                    out_pred = torch.clamp(out_pred, min = 0, max = 1)
                    mse, psnr, ssim, lpips = calculate_metrics(out_pred, out_gt, lpips_dist, args)

                    batch_size_current = valid_data.shape[0]
                    valid_mse.update(np.mean(mse), batch_size_current)
                    valid_psnr.update(np.mean(psnr), batch_size_current)
                    valid_ssim.update(np.mean(ssim), batch_size_current)
                    valid_lpips.update(np.mean(lpips), batch_size_current)

            mse_min = valid_mse.avg if valid_mse.avg < mse_min else mse_min
            psnr_max = valid_psnr.avg if valid_psnr.avg > psnr_max else psnr_max
            ssim_max = valid_ssim.avg if valid_ssim.avg > ssim_max else ssim_max
            lpips_min = valid_lpips.avg if valid_lpips.avg < lpips_min else lpips_min

            elapsed_time = time.time() - start_time; start_time = time.time()
            print('******** iter [{}] / epoch [{:.4f}] / loss [{:.4f}] ********'
                      .format(train_i+1, (train_i+1)/len(trainloader), train_loss.avg))
            print('[current] mse: {:.3f}, psnr: {:.3f}, ssim: {:.3f}, lpips: {:.3f}'
                      .format(valid_mse.avg, valid_psnr.avg, valid_ssim.avg, valid_lpips.avg))
            print('[  best ] mse: {:.3f}, psnr: {:.3f}, ssim: {:.3f}, lpips: {:.3f}'
                      .format(mse_min, psnr_max, ssim_max, lpips_min))
            print('elapsed time: {:.0f} sec'.format(elapsed_time))
            train_loss.reset(); valid_mse.reset(); valid_psnr.reset(); valid_ssim.reset(); valid_lpips.reset()

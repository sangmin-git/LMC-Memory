import torch

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(pred, gt, lpips_dist, args):
    batch_size = pred.shape[0]
    multi_channel = True if args.img_channel > 1 else False

    pred = pred.cpu().numpy()
    pred = np.transpose(pred, [0,1,3,4,2])
    gt = gt.cpu().numpy()
    gt = np.transpose(gt, [0,1,3,4,2])

    mse_mean = np.zeros(args.out_len, dtype=pred.dtype)
    psnr_mean = np.zeros(args.out_len, dtype=pred.dtype)
    ssim_mean = np.zeros(args.out_len, dtype=pred.dtype)
    lpips_mean = np.zeros(args.out_len, dtype=pred.dtype)
    gt = gt.astype(dtype=pred.dtype)

    for frame_i in range(-args.out_len, 0):
        for batch_i in range(batch_size):
            gt_frame = gt[batch_i,frame_i,:,:,:]
            pred_frame = pred[batch_i,frame_i,:,:,:]
            if args.img_channel == 1:
                gt_frame = np.squeeze(gt_frame)
                pred_frame = np.squeeze(pred_frame)
            mse_mean[frame_i] += mean_squared_error(gt_frame, pred_frame)/batch_size
            psnr_mean[frame_i] += peak_signal_noise_ratio(gt_frame, pred_frame)/batch_size
            ssim_mean[frame_i] += structural_similarity(gt_frame, pred_frame, multichannel=multi_channel)/batch_size

        batch_gt = gt[:,frame_i,:,:,:]
        batch_pred = pred[:,frame_i,:,:,:]
        batch_gt = np.transpose(batch_gt, [0,3,1,2])
        batch_pred = np.transpose(batch_pred, [0,3,1,2])
        if args.img_channel == 1:
            batch_gt = np.repeat(batch_gt, 3, axis=1)
            batch_pred = np.repeat(batch_pred, 3, axis=1)
        batch_gt = torch.from_numpy(batch_gt).float().to('cuda')
        batch_pred = torch.from_numpy(batch_pred).float().to('cuda')
        lpips_mean[frame_i] += np.mean(lpips_dist(batch_gt, batch_pred).cpu().numpy())

    mse_mean *= args.img_size**2

    return mse_mean, psnr_mean, ssim_mean, lpips_mean

from torch.utils.data import Dataset

import cv2
import os
import numpy as np


def make_dataset(dataset_dir):
    frame_path = []
    # Find and loop over all the clips in root `dir`.
    for index, folder in enumerate(sorted(os.listdir(dataset_dir))):
        clipsFolderPath = os.path.join(dataset_dir, folder)
        # Skip items which are not folders.
        if not (os.path.isdir(clipsFolderPath)):
            continue
        frame_path.append([])
        # Find and loop over all the frames inside the clip.
        for image in sorted(os.listdir(clipsFolderPath)):
            # Add path to list.
            frame_path[index].append(os.path.join(clipsFolderPath, image))
    return frame_path

class MovingMNIST(Dataset):
    def __init__(self, dataset_dir, seq_len, train=True):
        self.frame_path = make_dataset(dataset_dir)
        self.seq_len = seq_len
        self.train = train

        self.clips = []
        for video_i in range(len(self.frame_path)):
            video_frame_num = len(self.frame_path[video_i])
            self.clips += [(video_i, t) for t in range(video_frame_num - seq_len)] if train \
                else [(video_i, t * seq_len) for t in range(video_frame_num // seq_len)]

    def __getitem__(self, idx):
        (video_idx, data_start) = self.clips[idx]
        sample = []
        for frame_range_i in range(data_start, data_start+self.seq_len):
            frame = cv2.imread(self.frame_path[video_idx][frame_range_i], cv2.IMREAD_GRAYSCALE)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32)
            frame = frame/255.0
            sample.append(frame)
        return sample

    def __len__(self):
        return len(self.clips)

class KTH(Dataset):
    def __init__(self, dataset_dir, seq_len, train=True):
        self.frame_path = make_dataset(dataset_dir)
        self.seq_len = seq_len
        self.train = train

        self.clips = []
        for video_i in range(len(self.frame_path)):
            video_frame_num = len(self.frame_path[video_i])
            self.clips += [(video_i, t) for t in range(video_frame_num - seq_len)] if train \
                else [(video_i, t * seq_len) for t in range(video_frame_num // seq_len)]

    def __getitem__(self, idx):
        (video_idx, data_start) = self.clips[idx]
        sample = []
        for frame_range_i in range(data_start, data_start+self.seq_len):
            frame = cv2.imread(self.frame_path[video_idx][frame_range_i], cv2.IMREAD_GRAYSCALE)
            frame = np.expand_dims(frame, axis=0)
            frame = frame.astype(np.float32)
            frame = frame/255.0
            sample.append(frame)
        return sample

    def __len__(self):
        return len(self.clips)


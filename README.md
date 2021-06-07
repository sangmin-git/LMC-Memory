## Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning 


<div align="center"><img width="98%" src="https://user-images.githubusercontent.com/41602474/112792595-bc1a3a00-909e-11eb-9d7c-9890fdb2b254.PNG" /></div>


> 
This repository contains the official PyTorch implementation of the following paper:
> **Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning (CVPR 2021 Oral)**<br>
> Sangmin Lee, Hak Gu Kim, Dae Hwi Choi, Hyung-Il Kim, and Yong Man Ro<br>
> Paper: https://arxiv.org/abs/2104.00924<br>
> 
> **Abstract** *Our work addresses long-term motion context issues for predicting future frames. To predict the future precisely, it is required to capture which long-term motion context (e.g., walking or running) the input motion (e.g., leg movement) belongs to. The bottlenecks arising when dealing with the long-term motion context are: (i) how to predict the long-term motion context naturally matching input sequences with limited dynamics, (ii) how to predict the long-term motion context with high-dimensionality (e.g., complex motion). To address the issues, we propose novel motion context-aware video prediction. To solve the bottleneck (i), we introduce a long-term motion context memory (LMC-Memory) with memory alignment learning. The proposed memory alignment learning enables to store long-term motion contexts into the memory and to match them with sequences including limited dynamics. As a result, the long-term context can be recalled from the limited input sequence. In addition, to resolve the bottleneck (ii), we propose memory query decomposition to store local motion context (i.e., low-dimensional dynamics) and recall the suitable local context for each local part of the input individually. It enables to boost the alignment effects of the memory. Experimental results show that the proposed method outperforms other sophisticated RNN-based methods, especially in long-term condition. Further, we validate the effectiveness of the proposed network designs by conducting ablation studies and memory feature analysis. The source code of this work is available.*

## Preparation

### Requirements
- python 3
- pytorch 1.6+
- opencv-python
- scikit-image
- lpips
- numpy

### Datasets
This repository supports Moving-MNIST and KTH-Action datasets. 
- [Moving-MNIST](https://github.com/jthsieh/DDPAE-video-prediction/blob/master/data/moving_mnist.py)
- [KTH-Action](https://www.csc.kth.se/cvap/actions/)

After obtaining the datasets, preprocess the data as image files (refer to below). 
```shell
# Dataset preparation example:
movingmnist
├── train
│   ├── video_00000
│   │   ├── frame_00000.jpg
...
│   │   ├── frame_xxxxx.jpg
...
│   ├── video_xxxxx
```

## Training the Model
`train.py` saves the weights in `--checkpoint_save_dir` and shows the training logs.

To train the model, run following command:
```shell
# Training example for Moving-MNIST
python train.py \
--dataset 'movingmnist' \
--train_data_dir 'enter_the_path' --valid_data_dir 'enter_the_path' \
--checkpoint_save_dir './checkpoints' \
--img_size 64 --img_channel 1 --memory_size 100 \
--short_len 10 --long_len 30 --out_len 30 \
--batch_size 128 --lr 0.0002 --iterations 300000
```
```shell
# Training example for KTH-Action
python train.py \
--dataset 'kth' \
--train_data_dir 'enter_the_path' --valid_data_dir 'enter_the_path' \
--checkpoint_save_dir './checkpoints' \
--img_size 128 --img_channel 1 --memory_size 100 \
--short_len 10 --long_len 40 --out_len 40 \
--batch_size 32 --lr 0.0002 --iterations 300000
```
Descriptions of training parameters are as follows:
- `--dataset`: training dataset (movingmnist or kth)
- `--train_data_dir`: directory of training set  `--valid_data_dir`: directory of validation set
- `--checkpoint_save_dir`: directory for saving checkpoints
- `--img_size`: height and width of frame  `--img_channel`: channel of frame  `--memory_size`: memory slot size
- `--short_len`: number of short frames  `--long_len`: number of long frames  `--out_len`: number of output frames
- `--batch_size`: mini-batch size  `--lr`: learning rate  `--iterations`: number of total iterations
- Refer to `train.py` for the other training parameters

## Testing the Model
`test.py` saves the predicted frames in `--test_result_dir` or evalute the performances.

To test the model, run following command:
```shell
# Testing example for Moving-MNIST
python test.py \
--dataset 'movingmnist' --make_frame True \
--test_data_dir 'enter_the_path' --test_result_dir 'enter_the_path' \
--checkpoint_load_file 'enter_the_path' \
--img_size 64 --img_channel 1 --memory_size 100 \
--short_len 10 --out_len 30 \
--batch_size 8
```
```shell
# Testing example for KTH-Action
python test.py \
--dataset 'kth' --make_frame True \
--test_data_dir 'enter_the_path' --test_result_dir 'enter_the_path' \
--checkpoint_load_file 'enter_the_path' \
--img_size 128 --img_channel 1 --memory_size 100 \
--short_len 10 --out_len 40 \
--batch_size 8
```
Descriptions of testing parameters are as follows:
- `--dataset`: test dataset (movingmnist or kth)  `--make_frame`: whether to generate predicted frames
- `--test_data_dir`: directory of test set  `--test_result_dir`: directory for saving predicted frames
- `--checkpoint_load_file`: file path for loading checkpoint
- `--img_size`: height and width of frame  `--img_channel`: channel of frame  `--memory_size`: memory slot size
- `--short_len`: number of short frames  `--out_len`: number of output frames
- `--batch_size`: mini-batch size
- Refer to `test.py` for the other testing parameters

## Pretrained Models
You can download the pretrained models.
- [Pretrained model for Moving-MNIST](https://www.dropbox.com/s/c2yl2f7znzmj8mf/trained_file_movingmnist.pt?dl=0)
- [Pretrained model for KTH-Action](https://www.dropbox.com/s/nt015y70moqgy76/trained_file_kth.pt?dl=0)

## Citation
If you find this work useful in your research, please cite the paper:
```
@inproceedings{lee2021video,
  title={Video Prediction Recalling Long-term Motion Context via Memory Alignment Learning},
  author={Lee, Sangmin and Kim, Hak Gu and Choi, Dae Hwi and Kim, Hyung-Il and Ro, Yong Man},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 4
# Current configuration parameter method
mode = "valid"
# Experiment name, easy to save weights and log files
exp_name = "fsrcnn_x4"

if mode == "train":
    # Dataset
    train_image_dir = f"data/T91/FSRCNN/train"
    valid_image_dir = f"data/T91/FSRCNN/valid"
    test_lr_image_dir = f"data/Set5/LRbicx{upscale_factor}"
    test_hr_image_dir = f"data/Set5/GTmod12"

    image_size = 20
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""

    # Total number of epochs
    epochs = 3000

    # SGD optimizer parameter
    model_lr = 1e-3
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    print_frequency = 200

if mode == "valid":
    # Test data address
	#记得上面upscale_factor
    lr_dir = f"../DIV2K/DIV2K_valid_LR_bicubic/X4"
    sr_dir = f"../Output/DIV2K_X4/FSRCNN/results/SR2"
    hr_dir = f"../DIV2K/DIV2K_valid_HR"

    #model_path = f"results/{exp_name}/best.pth.tar"
    model_path = f"results/fsrcnn_x4.pth.tar"

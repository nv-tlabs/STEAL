# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import numpy as np
import torch.utils.data
from utils import dataloader
from models.casenet import casenet101 as CaseNet101
import os
import cv2
import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir_val', type=str, default='./data/sbd/data_aug/')
parser.add_argument('--flist_val', type=str, default='./data/sbd/data_aug/val_list.txt')

parser.add_argument('--ckpt',
                    type=str,
                    default='./checkpoints/sbd/model_checkpoint.pt')

parser.add_argument('--output_folder', type=str,
                    default='./output/sbd/')
parser.add_argument('--dataset', type=str, default='sbd')

parser.add_argument('--n_classes', type=int, default=20)


def save_pred(im_info, predictions, num_cls, output_dir):
    org_height, org_width = im_info['orig_size']
    filename = os.path.basename(im_info['impath'][0])
    img_result_name = os.path.splitext(filename)[0] + '.png'
    for idx_cls in range(num_cls):
        score_pred = predictions.data.cpu().numpy()[0][idx_cls, 0:org_height, 0:org_width]
        im = (score_pred * 255).astype(np.uint8)
        result_root = os.path.join(output_dir, 'class_' + str(idx_cls + 1))
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        cv2.imwrite(
            os.path.join(result_root, img_result_name),
            im)


def do_test_sbd(net_, val_data_loader_, cuda, output_folder, n_classes):
    print('Running Inference....')
    net_.eval()
    output_images_dir = os.path.join(output_folder, 'val_images')

    if not os.path.isdir(output_images_dir):
        os.makedirs(output_images_dir)

    for i_batch, (im_info, input_img, input_gt) in tqdm.tqdm(enumerate(val_data_loader_), total=len(val_data_loader_)):
        if cuda:
            im = input_img.cuda(async=True)
        else:
            im = input_img

        out_masks = net_(im)

        prediction = torch.sigmoid(out_masks[0])
        save_pred(im_info, prediction, n_classes, output_images_dir)

    return 0


def main():
    args = parser.parse_args()
    print('****')
    print(args)
    print('****')

    output_folder = args.output_folder
    root_dir_val = args.root_dir_val
    flist_val = args.flist_val
    ckpt = args.ckpt

    # --
    n_classes = args.n_classes
    crop_size_val = 512

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    net = CaseNet101()
    net = torch.nn.DataParallel(net.cuda())

    # val set
    val_dataset = dataloader.ValidationDataset(root_dir_val, flist_val, n_classes=n_classes, crop_size=crop_size_val)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=1,
                                                  shuffle=False)

    print('loading ckpt :%s' % ckpt)
    net.load_state_dict(torch.load(ckpt), strict=True)

    do_test_sbd(net, val_data_loader, cuda=True, n_classes=n_classes, output_folder=output_folder)


if __name__ == '__main__':
    main()

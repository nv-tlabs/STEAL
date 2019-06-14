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
from models.casenet import casenet101 as CaseNet101

from utils.dataloader import default_flist_reader
import os
import cv2
import tqdm

import argparse

#For fair comparison, this is inspired by the way CASENET inference procedure works
def do_test(net, output_folder, test_lst, n_classes=19, image_h=1024, image_w=2048, patch_h=512, patch_w=512,
            step_size_y=256, step_size_x=256, pad=16):
    num_cls = n_classes
    # image_h = 1024  # Need to pre-determine test image size
    # image_w = 2048  # Need to pre-determine test image size

    net.eval()
    if output_folder is not None:
        output_images_dir_iter = os.path.join(output_folder, 'val_images')
        if not os.path.isdir(output_images_dir_iter):
            os.makedirs(output_images_dir_iter)

    if ((2 * pad) % 8) != 0:
        raise ValueError('Pad number must be able to be divided by 8!')
    step_num_y = (image_h - patch_h + 0.0) / step_size_y
    step_num_x = (image_w - patch_w + 0.0) / step_size_x

    if round(step_num_y) != step_num_y:
        raise ValueError('Vertical sliding size can not be divided by step size!')

    if round(step_num_x) != step_num_x:
        raise ValueError('Horizontal sliding size can not be divided by step size!')

    step_num_y = int(step_num_y)
    step_num_x = int(step_num_x)
    mean_value = (104.008, 116.669, 122.675)  # BGR

    pred_set = []  # only used if output_folder is none.
    for idx_img in tqdm.tqdm(range(len(test_lst))):
        in_ = cv2.imread(test_lst[idx_img]).astype(np.float32)
        width, height, chn = in_.shape[1], in_.shape[0], in_.shape[2]
        im_array = cv2.copyMakeBorder(in_, pad, pad, pad, pad, cv2.BORDER_REFLECT)

        if (height != image_h) or (width != image_w):
            raise ValueError('Input image size must be' + str(image_h) + 'x' + str(image_w) + '!')

        # Perform patch-by-patch testing
        score_pred = np.zeros((height, width, num_cls))
        mat_count = np.zeros((height, width, 1))
        for i in range(0, step_num_y + 1):
            offset_y = i * step_size_y
            for j in range(0, step_num_x + 1):
                offset_x = j * step_size_x

                # crop overlapped regions from the image
                in_ = np.array(
                    im_array[offset_y:offset_y + patch_h + 2 * pad, offset_x:offset_x + patch_w + 2 * pad, :])
                in_ -= np.array(mean_value)
                in_ = in_.transpose((2, 0, 1))  # HxWx3 -> 3xHxW
                # ---
                in_ = torch.from_numpy(in_).cuda()
                in_ = in_.unsqueeze(dim=0)  # 1x3xHxW

                out_masks = net(in_)  #
                prediction = torch.sigmoid(out_masks[0]).data.cpu().numpy()[0]
                # add the prediction to score_pred and increase count by 1
                score_pred[offset_y:offset_y + patch_h, offset_x:offset_x + patch_w, :] += \
                    np.transpose(prediction, (1, 2, 0))[pad:-pad, pad:-pad, :]
                mat_count[offset_y:offset_y + patch_h, offset_x:offset_x + patch_w, 0] += 1.0

        score_pred = np.divide(score_pred, mat_count)

        img_base_name = os.path.basename(test_lst[idx_img])
        img_result_name = os.path.splitext(img_base_name)[0] + '.png'
        if output_folder is None:
            pred_set.append(score_pred)
            continue

        for idx_cls in range(num_cls):
            im = (score_pred[:, :, idx_cls] * 255).astype(np.uint8)
            result_root = os.path.join(output_images_dir_iter, 'class_' + str(idx_cls + 1))
            if not os.path.exists(result_root):
                os.makedirs(result_root)
            cv2.imwrite(
                os.path.join(result_root, img_result_name),
                im)

            # scaling 50% resolution for fast evaluation.
            im_small = cv2.resize(im, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            result_root = os.path.join(output_images_dir_iter + '_scaled_0_5', 'class_' + str(idx_cls + 1))
            if not os.path.exists(result_root):
                os.makedirs(result_root)

            cv2.imwrite(
                os.path.join(result_root, img_result_name),
                im_small)

    return pred_set  # empty if output_folder exits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir_val', type=str,
                        default='./data/cityscapes-preprocess/data_proc')

    parser.add_argument('--flist_val', type=str,
                        default='./data_proc/val.txt')

    # --
    parser.add_argument('--ckpt', type=str, default='./checkpoints/cityscapes/model_checkpoint.pt')

    parser.add_argument('--output_folder', type=str, default='./output/cityscapes')

    args = parser.parse_args()
    print('****')
    print(args)
    print('****')

    output_folder = args.output_folder
    root_dir_val = args.root_dir_val
    flist_val = args.flist_val
    ckpt = args.ckpt

    # ---

    net = CaseNet101(nclasses=19)
    net = torch.nn.DataParallel(net.cuda())

    print('loading ckpt %s...' % ckpt)
    net.load_state_dict(torch.load(ckpt), strict=True)

    imlist = default_flist_reader(None, flist_val)
    test_lst = [os.path.join(root_dir_val, im_path) for (im_path, _) in imlist]

    do_test(net, output_folder, test_lst, n_classes=19)


if __name__ == '__main__':
    main()

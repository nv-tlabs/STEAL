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


import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
import cv2


# from torchvision import transforms, datasets
# Image file list reader function taken from https://github.com/pytorch/vision/issues/81

def default_flist_reader(root, flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            splitted = line.strip().split()
            if len(splitted) == 2:
                impath, imlabel = splitted
            elif len(splitted) == 1:
                impath, imlabel = splitted[0], None
            else:
                raise ValueError('weird length ?')
            impath = impath.strip('../')
            imlist.append((impath, imlabel))

    return imlist


def _is_bit_set(x, n):
    if x & (1 << n):
        return 1
    else:
        return 0


def _decode_integer(_int, channels, output_):
    for c in range(channels):
        output_[c] = _is_bit_set(_int, c)


def binary_file_to_channel_masks(bin_file, h, w, channels, seen_classes=None, ignore_pixel_id_map=(31, 255)):
    array = np.fromfile(bin_file, dtype=np.uint32)
    # i'll asssume this array is very sparse so this should be fast.
    idxs = np.argwhere(array != 0)
    arr_chn = np.zeros((array.shape[0], channels))
    #
    for idx in idxs:
        for c in range(channels):
            ignore_pixel = _is_bit_set(array[idx], ignore_pixel_id_map[0]) == 1
            if ignore_pixel is True:
                # print('ignoring pixel')
                arr_chn[idx, c] = ignore_pixel_id_map[1]  # 255
            else:
                arr_chn[idx, c] = _is_bit_set(array[idx], c)
            # for debug?
            if seen_classes is not None:
                if arr_chn[idx, c] != 0:
                    seen_classes.add(c)
    return arr_chn.reshape(h, w, channels)


def seg_img_to_Kchannels(imfile, klasses):
    image = np.array(Image.open(imfile))
    masks = [image == (i + 1) for i in range(klasses)]
    # TODO maybe this need to handle ignore pixels at some point.
    return np.stack(masks, axis=0).astype(np.uint8)


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, n_classes, transform, shuffle_list, ref_gt=False, read_seg=False, *args, **kwargs):
        self.root = root
        self.imlist = default_flist_reader(root, flist)
        if shuffle_list is True:
            np.random.shuffle(self.imlist)

        self.transform = transform
        self.n_classes = n_classes
        self.ignore_value = 255
        self._compute_ref_gt = ref_gt
        self._read_seg = read_seg

    def _read_image(self, input_image_path, *args):
        raise NotImplementedError()

    def _read_gt(self, gtpath, *args):
        raise NotImplementedError()

    def _ref_gt(self, gt):
        """

        :param gt: C,H,W
        :return:
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        impath, gtpath = (
            os.path.join(self.root, *self.imlist[index][0].split('/')),
            os.path.join(self.root, *self.imlist[index][1].split('/'))
        )
        image = self._read_image(impath)
        width, height = Image.open(impath).size
        gt = self._read_gt(gtpath, (height, width))  # reading only the header..faster
        image_info = {'impath': impath, 'gtpath': gtpath, 'orig_size': (height, width)}
        return image_info, image, gt

    def __len__(self):
        return len(self.imlist)


class ValidationDataset(ImageFilelist):
    def __init__(self, root, flist, n_classes, transform=None, crop_size=512):
        super(ValidationDataset, self).__init__(root, flist, n_classes, transform, shuffle_list=False)
        self._crop_size = crop_size

    def _read_image(self, input_image_path, *args):
        crop_size = self._crop_size
        mean_value = (104.008, 116.669, 122.675)  # BGR
        original_im = cv2.imread(input_image_path).astype(np.float32)
        in_ = original_im
        width, height = in_.shape[1], in_.shape[0]
        if crop_size < width or crop_size < height:
            raise ValueError('Input image size must be smaller than crop size!')
        elif crop_size == width and crop_size == height:
            # ("WARNING *** skipping because of crop_size ")
            pass
        else:
            pad_x = crop_size - width
            pad_y = crop_size - height
            in_ = cv2.copyMakeBorder(in_, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=mean_value)
        in_ -= np.array(mean_value)
        in_ = in_.transpose((2, 0, 1))  # HxWx3 -> 3xHxW
        return in_

    def _read_gt(self, gtpath, size):
        gt_mask = binary_file_to_channel_masks(gtpath, size[0], size[1], self.n_classes, None)
        crop_size = self._crop_size
        width, height = gt_mask.shape[1], gt_mask.shape[0]
        if crop_size < width or crop_size < height:
            raise ValueError('Input gt size must be smaller than crop size!')
        elif crop_size == width and crop_size == height:
            # ("WARNING GT *** skipping because of crop_size ")
            pass
        else:
            pad_x = crop_size - width
            pad_y = crop_size - height
            gt_mask = cv2.copyMakeBorder(gt_mask, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT,
                                         value=[self.ignore_value] * 4)

        return np.transpose(gt_mask, [2, 0, 1])



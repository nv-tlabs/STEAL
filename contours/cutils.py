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

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import torch


def update_callback_in_image(image):
    # this is for debugging so i am importing here to keep it clean
    import matplotlib.pyplot as plt
    def fn_post_process_callback(evol, pxwise):
        colors = ('r', 'y', 'g', 'b', 'w')
        if image is not None:
            plt.imshow(image)
        ctrs_l = []
        ctrs_labels = []
        for i in range(len(evol)):
            cntr = plt.contour(evol[i], [0.5], colors=colors[i], lw=2)
            h1, _ = cntr.legend_elements()
            ctrs_l.append(h1[0])
            ctrs_labels.append('contour: %i' % i)

        plt.legend(ctrs_l, ctrs_labels)
        plt.show()
        return pxwise

    return fn_post_process_callback


def seg2edges(image, radius, label_ignores=(255,)):
    """
    :param image: semantic map should be HxWx1 with values 0,1,label_ignores
    :param radius: radius size
    :param label_ignores: values to mask.
    :return: edgemap with boundary computed based on radius
    """
    if radius < 0:
        return image

    ignore_dict = {}

    for ignore_id in label_ignores:
        idxs = np.nonzero(image == ignore_id)
        image[idxs] = 0.
        ignore_dict[ignore_id] = idxs

    # we need to pad the borders, to solve problems with dt around the boundaries of the image.
    image_pad = np.pad(image, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    dist1 = distance_transform_edt(image_pad)
    dist2 = distance_transform_edt(1.0 - image_pad)
    dist = dist1 + dist2

    # removing padding, it shouldnt affect result other than if the image is seg to the boundary.
    dist = dist[1:-1, 1:-1]
    assert dist.shape == image.shape

    dist[dist > radius] = 0

    dist = (dist > 0).astype(np.uint8)  # just 0 or 1

    ##bringing back the ignored areas back
    for k, v in ignore_dict.items():
        dist[v] = k

    return dist


def seg2edges_2d(image, radius):
    """
    :param image: semantic map should be CxHxW with values 0,1,label_ignores
    :param radius: radius size
    :param label_ignores: values to mask.
    :return: edgemap with boundary computed based on radius
    """

    # we need to pad the borders, to solve problems with dt around the boundaries of the image.
    image_pad = np.pad(image, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    dist1 = distance_transform_edt(image_pad)
    dist2 = distance_transform_edt(1.0 - image_pad)
    dist = dist1 + dist2

    # removing padding, it shouldnt affect result other than if the image is seg to the boundary.
    dist = dist[:, 1:-1, 1:-1]
    assert dist.shape == image.shape, 'dist_shape: %s ; image_shape:%s ' % (dist.shape, image.shape)

    dist[dist > radius] = 0

    dist = (dist > 0).astype(np.uint8)  # just 0 or 1

    return dist


def compute_h_additive(gt_K, pK_Image, lambda_, alpha):
    # normalizing pK_image so that's [0..1]
    pK_Image = pK_Image / (np.max(pK_Image) + 1e-5)

    gPimage = 1.0 / np.sqrt(1.0 + alpha * pK_Image)
    gpGT = 1.0 / np.sqrt(1.0 + alpha * gt_K)
    gTotal = gPimage + lambda_ * gpGT
    return gTotal


def compute_h_additive_torch(gt_K, pK_Image, lambda_, alpha):
    # normalizing pK_image so that's [0..1]
    pK_Image = pK_Image / (torch.max(pK_Image) + 1e-5)

    gPimage = 1.0 / torch.sqrt(1.0 + alpha * pK_Image)
    gpGT = 1.0 / torch.sqrt(1.0 + alpha * gt_K)
    gTotal = gPimage + lambda_ * gpGT
    return gTotal


def compute_h_caselles_torch(gt_K, pK_Image, lambda_, alpha):
    # normalizing pK_image so that's [0..1]
    pK_Image = pK_Image / (torch.max(pK_Image) + 1e-5)

    gPimage = 1.0 / (1.0 + alpha * pK_Image)
    gpGT = 1.0 / (1.0 + alpha * gt_K)
    gTotal = gPimage + lambda_ * gpGT
    return gTotal

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


import os
import numpy as np
from PIL import Image
import scipy.io
import random
import scipy.sparse

import scipy.ndimage.morphology as morp
import cv2
from skimage.measure import regionprops
from skimage import measure
from collections import OrderedDict


class InputReaderBase:
    def __init__(self, in_path, file_list, n_classes):
        """

        :param path:
        :param file_list:
        :param n_classes: tuple (start,end)
        """
        self._in_path = in_path
        self._file_list = file_list
        if file_list is not None:
            self._read_list = open(file_list, 'r').read().splitlines()
        else:
            self._read_list = None

        self.n_classes = n_classes
        self._seed = 123
        self._classes_to_keep = []
        self._remapping = OrderedDict()

    def set_classes_to_keep(self, classes_to_keep):
        self._classes_to_keep = classes_to_keep
        self._remapping = OrderedDict()

    def set_external_list(self, ext_list):
        self._read_list = ext_list

    def randompick(self, max_number, seed=123):
        self._seed = seed
        random.seed(seed)
        random.shuffle(self._read_list)
        self._file_list = 'external'
        self._read_list = self._read_list[:max_number]

    def _ignore_classes(self, masks):
        if len(self._classes_to_keep) == 0:
            return masks
        idx = 0
        result = []
        for i in range(len(masks)):
            if i in self._classes_to_keep:
                mask = masks[i]
                result.append(mask)
                self._remapping[idx] = i
                idx = idx + 1
        return np.array(result)

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        return len(self._read_list)


class InputReader(InputReaderBase):
    def __getitem__(self, item):
        stack = []
        filename = self._read_list[item]
        for idx_cls in range(self.n_classes[0], self.n_classes[1] + self.n_classes[0]):
            img_path = os.path.join(self._in_path, 'class_' + str(idx_cls), filename + '.png')
            img = np.array(Image.open(img_path)) / 255.0
            stack.append(img)
        assert len(stack) == self.n_classes[1], len(stack)
        stack = self._ignore_classes(stack)

        return filename, stack


class InputReaderBaseName(InputReaderBase):
    def __getitem__(self, item):
        stack = []
        filename = self._read_list[item]
        filename = os.path.basename(filename).split('.png')[0]
        for idx_cls in range(self.n_classes[0], self.n_classes[1] + self.n_classes[0]):
            img_path = os.path.join(self._in_path, 'class_' + str(idx_cls), filename + '.png')
            img = np.array(Image.open(img_path)) / 255.0
            stack.append(img)
        assert len(stack) == self.n_classes[1], len(stack)
        stack = self._ignore_classes(stack)

        return filename, stack


class InputReaderDummy(InputReaderBase):
    def __getitem__(self, item):
        filename = self._read_list[item]
        return filename, [None for _ in range(self.n_classes[1])]


class InputReaderRGBImage(InputReaderBase):
    def __getitem__(self, item):
        filename = self._read_list[item]
        img_path = os.path.join(self._in_path, 'img', filename + '.png')

        if not os.path.isfile(img_path):
            img_path = os.path.join(self._in_path, 'img', filename + '.jpg')

        img = Image.open(img_path)
        return filename, img


class InputReaderSemMat(InputReaderBase):
    @staticmethod
    def map2Kchannels(maps, klasses):
        masks = [maps == (i + 1) for i in range(klasses)]
        return np.stack(masks, axis=0).astype(np.uint8)

    def __getitem__(self, item):
        filename = self._read_list[item]
        mat_path = os.path.join(self._in_path, filename + '.mat')
        matlab = scipy.io.loadmat(mat_path)
        segmap = matlab['GTcls']['Segmentation'][0][0]
        masks = self.map2Kchannels(segmap, self.n_classes[1])
        assert len(masks) == self.n_classes[1], len(masks)
        masks = self._ignore_classes(masks)

        return filename, masks


class InputReaderSemMat2(InputReaderSemMat):
    def __getitem__(self, item):
        filename = self._read_list[item]
        mat_path = os.path.join(self._in_path, 'cls', filename + '.mat')
        matlab = scipy.io.loadmat(mat_path)
        segmap = matlab['GTcls']['Segmentation'][0][0]
        masks = self.map2Kchannels(segmap, self.n_classes[1])
        assert len(masks) == self.n_classes[1], len(masks)
        masks = self._ignore_classes(masks)
        return filename, masks


class InputReaderSemMatDemo(InputReaderSemMat):
    def __getitem__(self, item):
        mat_path = self._read_list[item]
        matlab = scipy.io.loadmat(mat_path)
        segmap = matlab['GTcls']['Segmentation'][0][0]
        masks = self.map2Kchannels(segmap, self.n_classes[1])
        assert len(masks) == self.n_classes[1], len(masks)
        masks = self._ignore_classes(masks)
        return os.path.basename(mat_path), masks


class InputReaderSemMat2BaseName(InputReaderSemMat):
    def __getitem__(self, item):
        filename = self._read_list[item]
        filename = os.path.basename(filename).split('.png')[0]

        mat_path = os.path.join(self._in_path, 'cls', filename + '.mat')
        matlab = scipy.io.loadmat(mat_path)
        segmap = matlab['GTcls']['Segmentation'][0][0]
        masks = self.map2Kchannels(segmap, self.n_classes[1])
        assert len(masks) == self.n_classes[1], len(masks)
        masks = self._ignore_classes(masks)
        return filename, masks


class InputReaderSemMatCoarse(InputReaderBase):
    def __init__(self, in_path, file_list, n_classes, delta):
        super(InputReaderSemMatCoarse, self).__init__(in_path, file_list, n_classes)
        self._delta = delta

    def map2Kchannels(self, maps, klasses):
        masks = []
        idx = 0
        for i in range(klasses):
            if (len(self._classes_to_keep) == 0) or (i in self._classes_to_keep):
                mask = (maps == (i + 1))
                coarse_mask = self._simulate_coarse_label(mask, self._delta)
                masks.append(coarse_mask)
                self._remapping[idx] = i
                idx = idx + 1

        return np.stack(masks, axis=0).astype(np.uint8)

    def _simulate_coarse_label(self, mask, delta, return_polys=False):
        all_zeros = not np.any(mask)
        if all_zeros:
            return mask

        erosion_iter = delta // 2
        eroded_mask = morp.binary_erosion(mask, iterations=erosion_iter)

        if not np.any(eroded_mask):
            eroded_mask = eroded_mask.astype(np.uint8)
            # if the objects are too small and the delta (erosion) is to big, the objects may disapear
            # if all disappear
            # so let's draw a circle
            properties = regionprops(mask.astype(np.int32))
            c_y, c_x = properties[0].centroid
            c_y = int(c_y)
            c_x = int(c_x)
            # --
            cv2.circle(eroded_mask, (c_x, c_y), 3, 1, -1)
            return eroded_mask

        polys = measure.find_contours(eroded_mask, 0)

        if len(polys) == 0:
            print('error getting poly..returning empty mask')
            return np.zeros_like(mask)

        final_mask = np.zeros_like(mask).astype(np.uint8).T
        final_mask = np.ascontiguousarray(final_mask)
        for poly in polys:
            result = cv2.approxPolyDP(poly.astype(np.int32), erosion_iter, True)[:, 0, :]
            cv2.fillPoly(final_mask, [result], 1)

        final_mask = final_mask.T

        if not np.any(final_mask):
            print('this mask ended up being empty')

        return final_mask

    def __getitem__(self, item):
        filename = self._read_list[item]
        mat_path = os.path.join(self._in_path, 'cls', filename + '.mat')
        matlab = scipy.io.loadmat(mat_path)
        segmap = matlab['GTcls']['Segmentation'][0][0]

        # this function inside does ignore classes... doing this way for speed
        masks = self.map2Kchannels(segmap, self.n_classes[1])

        return filename, masks


class InputReaderSemMatCoarsePerComponent(InputReaderSemMatCoarse):

    def getSim_fn(self):
        return None

    def _simulate_coarse_label(self, mask, delta):
        all_zeros = not np.any(mask)
        if all_zeros:
            return mask

        blobs_labels, n_blobs = measure.label(mask, background=0, return_num=True)

        simp_output = np.zeros_like(mask)

        fn_callback = self.getSim_fn()
        for blob_id in range(1, n_blobs + 1):
            mask_blob = (blobs_labels == blob_id)
            mask_blob = np.ascontiguousarray(mask_blob, np.uint8)

            if fn_callback is None:
                sim_blob = super(InputReaderSemMatCoarsePerComponent, self)._simulate_coarse_label(mask_blob, delta)
            else:
                sim_blob = fn_callback(mask_blob, delta)

            simp_output = simp_output + sim_blob

        #

        return simp_output


class InputReaderSemMatClickSim(InputReaderSemMatCoarsePerComponent):
    def _simulate_click(self, mask, delta):
        all_zeros = not np.any(mask)
        if all_zeros:
            return mask
        radius = 6
        sim_mask = np.zeros_like(mask)
        properties = regionprops(mask.astype(np.int32))
        c_y, c_x = properties[0].centroid

        c_y = c_y + delta * np.random.normal()
        c_x = c_x + delta * np.random.normal()

        c_y = max(c_y, 0)
        c_x = max(c_x, 0)

        c_y = int(c_y)
        c_x = int(c_x)
        # --
        cv2.circle(sim_mask, (c_x, c_y), delta, 1, -1)
        return sim_mask

    def getSim_fn(self):
        return self._simulate_click

    def _simulate_coarse_label(self, mask, delta):
        return super(InputReaderSemMatClickSim, self)._simulate_coarse_label(mask, delta)


class InputReaderBdryMat(InputReaderBase):
    def __getitem__(self, item):
        filename = self._read_list[item]
        mat_path = os.path.join(self._in_path, filename + '.mat')
        matlab = scipy.io.loadmat(mat_path)
        boundaries = matlab['GTcls']['Boundaries']
        masks = [scipy.sparse.csr_matrix.todense(boundaries[0][0][c_id][0]) for c_id in range(0, 20)]

        assert len(masks) == self.n_classes[1], len(masks)

        return filename, masks

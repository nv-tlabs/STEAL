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
import os
import skimage.measure as measure

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

debug_remote = False
if debug_remote is None:
    pass
elif debug_remote:
    plt.switch_backend('Qt5Agg')
else:
    plt.switch_backend('Agg')


class VisualizerBox:
    #
    def city_pallete(self):
        CITYSCAPE_PALLETE = np.asarray([
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
            [0, 0, 0]], dtype=np.uint8)
        self.colors_are_a_list = False
        return CITYSCAPE_PALLETE

    #
    def sbd_pallete(self):
        SBD_PALLETE = np.asarray([
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
            [0, 0, 0]], dtype=np.uint8)
        self.colors_are_a_list = False
        return SBD_PALLETE

    def css4_colors_pallete(self):
        self.colors_are_a_list = True
        return list(mcolors.CSS4_COLORS.keys())

    def css4_fushia(self):
        self.colors_are_a_list = True
        rrr = list(mcolors.CSS4_COLORS.keys())
        rrr[13] = 'fuchsia'  # forcing to this color, remove if need it.
        return rrr

    def __init__(self, dataset_color, plt_backend=None, fig_size=(8, 12), only_contour=False, postfix_as_name=True):
        self.colors_are_a_list = False
        if dataset_color == 'cityscapes':
            print('dataset color: cityscapes')
            self._mycolors = self.city_pallete()
        elif dataset_color == 'sbd':
            print('dataset color: sbd')
            self._mycolors = self.sbd_pallete()
        elif dataset_color == 'css4':
            print('dataset color: css4')
            self._mycolors = self.css4_colors_pallete()
        elif dataset_color == 'css4_fushia':
            print('dataset color: css4_fushia')
            self._mycolors = self.css4_fushia()
        else:
            raise ValueError()

        self._output_f = None
        self._fig_size = fig_size
        self._only_contour = only_contour
        self._postfix_as_name = postfix_as_name

        if plt_backend is not None:
            print('Switching backend to %s' % plt_backend)
            plt.switch_backend(plt_backend)

    def plot_multichannel_mask(self, ax, masks, remapping_dict=None, ref_contour=None):

        for i in range(masks.shape[0]):
            mask = masks[i]

            if not np.any(mask):
                continue

            contours = measure.find_contours(mask, 0)
            # TODO this is a copying and pasting hack, it can be done properly.
            if ref_contour is not None and self._only_contour is True:
                ref_contour_i = measure.find_contours(ref_contour[i], 0)
                for contour in ref_contour_i:
                    contour = np.fliplr(contour)
                    ax.plot(contour[:, 0], contour[:, 1], linewidth=3, color='red')  #

            for contour in contours:
                contour = np.fliplr(contour)
                if remapping_dict is None:
                    c_id = i
                else:
                    c_id = remapping_dict[i]
                if self._only_contour is False:
                    if self.colors_are_a_list:
                        color = self._mycolors[c_id]
                    else:
                        color = self._mycolors[c_id] / 255.0
                    p = patches.Polygon(contour, facecolor=color, edgecolor='white', linewidth=0,
                                        alpha=0.5)
                    ax.add_patch(p)
                    ax.plot(contour[:, 0], contour[:, 1], linewidth=2,
                            color='orange')  # #

                else:
                    # #
                    simple_color = 'greenyellow'
                    ax.plot(contour[:, 0], contour[:, 1], linewidth=2,
                            color=simple_color, alpha=1)  #
        return ax

    def add_vis_list(self, images_dict, background=None, remapping_dict=None, grid=True, merge_channels=True,
                     exec_fn=None, ref_contour=None):
        if merge_channels is False:
            return NotImplementedError()

        if grid is True:
            f, ax = plt.subplots(len(images_dict.keys()), 1, figsize=self._fig_size)
        else:
            f, ax = plt.subplots(1, 1, figsize=self._fig_size)

        if background is None:
            h, w = images_dict.items()[0].shape[1:3]  # C, H,W
            background = np.ones((h, w)) * 255

        for i, (title, image_array) in enumerate(images_dict.items()):

            # just in case it was lazy loaded (eg.PIL)
            image_array = np.array(image_array)
            if not (type(ax) is np.ndarray):
                curr_ax = ax
            elif len(ax) > 1:
                curr_ax = ax[i]
            else:
                curr_ax = ax
            curr_ax.set_axis_off()
            curr_ax.set_title(title)

            if image_array.shape[0] == 1:  #
                curr_ax.imshow(image_array[0])
                if exec_fn is not None and grid is False:
                    exec_fn(f, curr_ax, title)

                continue

            # setting the background, usually input image...
            curr_ax.imshow(np.array(background), alpha=1)

            self.plot_multichannel_mask(curr_ax, image_array, remapping_dict, ref_contour)

            if exec_fn is not None and grid is False:
                exec_fn(f, curr_ax, title)

        if exec_fn is not None and grid is True:
            exec_fn(f, ax, 'grid')

        return f, ax

    def set_output_folder(self, output_f):
        self._output_f = output_f
        if self._output_f is not None:
            if not os.path.isdir(self._output_f):
                os.makedirs(self._output_f)
        print('Vis Output Dir:', self._output_f)

    def save_vis(self, image_name, images_dict, background=None, remapping_dict=None, grid=True, auto_show=False,
                 ref_contour=None):
        def exec_callback(f, ax, title, **kwargs):
            assert self._output_f is not None
            title = title.replace(" ", "").lower()
            if self._postfix_as_name is True:
                fname = os.path.join(self._output_f, image_name + '_' + title + '.jpg')
            else:
                fname = os.path.join(self._output_f, image_name + '.jpg')
            f.tight_layout()
            f.savefig(fname)
            ax.cla()

        self.add_vis_list(images_dict, background, remapping_dict, grid, exec_fn=exec_callback, ref_contour=ref_contour)
        if auto_show:
            plt.show()

        plt.close()

    def visualize(self, images_dict, background=None, remapping_dict=None, grid=True, merge_channels=True):
        f, ax = self.add_vis_list(images_dict, background, remapping_dict, grid, merge_channels)
        plt.show()

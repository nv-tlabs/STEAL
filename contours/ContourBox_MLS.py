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


from contours.morph_snakes import morphological_geodesic_active_contour
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import multiprocessing as mp
from contours.cutils import seg2edges, update_callback_in_image, compute_h_additive
from contours.ContourBox import LevelSetAlignmentBase
import torch
import warnings

# this is a workaround to avoid creating/copying the two big arrays to a shared memory area for multi-cpu.
# it only works on POSIX-compliant OS (linux,OSX) and it assumes there is no need for a lock.
# this data is read only.
shared_mem_data = None


class MLS(LevelSetAlignmentBase):
    def _fill_inside(self, bdry, method='fill_holes'):

        if method == 'fill_holes':
            seg = binary_fill_holes(bdry)
            return seg
        else:
            raise ValueError('_fill_inside wrong method:%s' % method)

    def _eval_singleK(self, gt_K, pK_Image, step_ckpts, lambda_, alpha, smoothing,
                      render_radius, is_gt_semantic, weight_canvas=None, **kwargs):

        def store_evolution_in(lst):
            """Returns a callback function to store the evolution of the level sets in
            the given list.
            """

            def _store(x, i):
                if i in step_ckpts:
                    lst.append(np.copy(x))

            return _store

        # This way of checking mostly cares about speed. as I am assuming the whole GT is very sparse.

        all_zeros = not np.any(gt_K)
        # nothing to do
        if all_zeros:
            out = [gt_K for _ in range(len(step_ckpts))]
            return [out, out]

        # let's remove for now the ignore areas
        ignore_dict = {}
        #
        for ignore_id in self.ignore_labels:
            idxs = np.nonzero(gt_K == ignore_id)
            gt_K[idxs] = 0.
            ignore_dict[ignore_id] = idxs

        # let's check again for zeros.
        all_zeros = not np.any(gt_K)  #

        # nothing to do
        if all_zeros:
            # ignore areas back
            for k, v in ignore_dict.items():
                gt_K[v] = k
            out = [gt_K for _ in range(len(step_ckpts))]
            zero_pp = np.zeros_like(gt_K)
            out2 = [zero_pp for _ in range(len(step_ckpts))]
            return [out, out2]

        if is_gt_semantic is False:
            # filling inside_ to represent the curve
            # this may have problem with boundaries that are not closed(corners of the image dimension)
            # be careful!!
            init_ls = self._fill_inside(gt_K)
        else:
            init_ls = gt_K
            gt_K = seg2edges(gt_K, radius=render_radius, label_ignores=self.ignore_labels)

        if self.fn_debug is not None:
            self.fn_debug(init_ls, 'init_ls')

        # List with intermediate results to save the evolution
        evolution = []
        callback = store_evolution_in(evolution)

        if weight_canvas is not None:
            if 'merge_weight' in self.options_dict:
                if self.options_dict['merge_weight'] > 0:
                    pK_Image = pK_Image + weight_canvas

        h = self._compute_h(gt_K, pK_Image, lambda_, alpha)

        if self.fn_debug is not None:
            self.fn_debug(h, 'h')

        #
        n_iterations = step_ckpts[-1]  # equal to my last checkpoint

        if 'balloon' in kwargs:
            balloon = kwargs['balloon']
        else:
            balloon = 0

        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0

        morphological_geodesic_active_contour(h, n_iterations, init_ls,
                                              smoothing=smoothing, balloon=balloon,
                                              threshold=threshold,
                                              iter_callback=callback)

        pixel_wise_evol = [seg2edges(evol, radius=render_radius, label_ignores=self.ignore_labels) for evol in
                           evolution]

        if 0 in step_ckpts:
            # appending the original gt_K
            # pixel_wise_evol.append(gt_K)
            # evolution.append(init_ls)
            # this can be appended it at the end as above... saving the shifting of the array.
            # but I want to visualize
            pixel_wise_evol.insert(0, gt_K)
            evolution.insert(0, init_ls)
        else:
            pass
            # ...

        ##bringing back the ignored areas back
        for k, v in ignore_dict.items():
            for j in range(len(pixel_wise_evol)):
                pixel_wise_evol[j][v] = k

        if self.fn_post_process_callback is None:
            return [pixel_wise_evol, pixel_wise_evol]
        else:
            return [pixel_wise_evol, self.fn_post_process_callback(evolution, pixel_wise_evol)]

    #
    def process_batch_fn(self, args):
        i, K, gt, pk, weight_canvas = args
        gt = gt[i]
        pk = pk[i]
        weightk_canvas = weight_canvas[i]

        gt_hat = []
        gt_hat_pp = []

        for j in range(K):
            gtk = gt[j]
            gt_hat_k, gt_hat_pp_k = self._eval_singleK(gtk, pk[j], weight_canvas=weightk_canvas[j], **self.options_dict)
            gt_hat.append(gt_hat_k)

            if self.fn_post_process_callback is None:
                gt_hat_pp.append(None)
            else:
                gt_hat_pp.append(gt_hat_pp_k)

        return i, [np.stack(gt_hat, axis=0), np.stack(gt_hat_pp, axis=0)]  # KxLStepsxHxW

    def process_batch_hack_multicpu(self, args):
        i, K, mem_id_gt, mem_id_pk, mem_id_weight_canvas = args

        if shared_mem_data is None:
            raise ValueError()

        gt, pk, weight_canvas = shared_mem_data[0], shared_mem_data[1], shared_mem_data[2]

        if mem_id_gt != id(gt):
            raise ValueError('error seems the memory id are not the same, not shared array... is this linux?')

        if mem_id_pk != id(pk):
            raise ValueError('error seems the memory id are not the same, not shared array... is this linux?')

        if mem_id_weight_canvas != id(weight_canvas):
            raise ValueError('error seems the memory id are not the same, not shared array... is this linux?')

        # seems all is fine...let's do it.
        args2 = (i, K, gt, pk, weight_canvas)
        return self.process_batch_fn(args2)

    def _multi_cpu_call(self, gt, pk):
        assert gt.shape == pk.shape
        global shared_mem_data
        shared_mem_data = (gt, pk)
        N, K, H, W = pk.shape
        #
        pool = mp.Pool(min(N, self.n_workers))

        output_ = pool.map(self.process_batch_hack_multicpu, [(i, K, id(gt), id(pk)) for i in range(N)])

        pool.close()
        pool.join()

        return output_

    def _multi_cpu_call_2(self, gt, pk, weight_canvas=None):
        """
        -->...similar to above but the workers can be reused in N*K elements instead of just K
        :param gt:
        :param pk:
        :return:
        """

        assert gt.shape == pk.shape
        N, K, H, W = pk.shape

        gt = np.reshape(gt, [N * K, 1, H, W])
        pk = np.reshape(pk, [N * K, 1, H, W])

        if weight_canvas is not None:
            weight_canvas = np.reshape(weight_canvas, [N * K, 1, H, W])
        else:
            weight_canvas = np.zeros_like(pk)

        global shared_mem_data
        shared_mem_data = (gt, pk, weight_canvas)

        #
        pool = mp.Pool(min(N * K, self.n_workers))

        output_ = pool.map(self.process_batch_hack_multicpu,
                           [(i, 1, id(gt), id(pk), id(weight_canvas)) for i in range(N * K)])

        pool.close()
        pool.join()
        # we need to reorder the array to make it compatible with the rest of the api.

        return output_

    def merge_everyone_but_me(self, pks_, me):
        canvas = np.zeros_like(pks_[0])
        for k in range(pks_.shape[0]):
            if k == me:
                continue
            canvas += pks_[k]

        canvas = canvas / (np.max(canvas) + 1e-6)
        return canvas

    def weigthing_merge(self, weight, pks):
        N, K, H, W = pks.shape
        output = np.zeros_like(pks)
        for n in range(N):
            item = pks[n]
            for k in range(K):
                # TODO this is quadratic for now in k :((, do properly.
                canvas = self.merge_everyone_but_me(item, k)
                output[n, k] = weight * canvas
        return output

    def __call__(self, gt_dict, pk):

        if not isinstance(gt_dict, dict):
            raise ValueError('wrong param')

        gt = gt_dict['seg']
        if isinstance(gt, torch.Tensor):
            gt = gt.byte().cpu().float().numpy()

        if isinstance(pk, torch.Tensor):
            pk = pk.cpu().numpy()

        assert gt.shape == pk.shape
        N, K, H, W = pk.shape

        weight_canvas = None
        if 'merge_weight' in self.options_dict:
            weights = self.options_dict['merge_weight']
            if weights >= 0:
                weight_canvas = self.weigthing_merge(weights, pk)
            elif weights == 0:
                print('skipping...')
            else:
                raise ValueError('negative ?, this shouldnt happen')

        if N * K > 1 and self.n_workers > 1:
            output_ = self._multi_cpu_call_2(gt, pk, weight_canvas)
        else:
            output_ = []
            for i in range(N):
                idx, gt_hat = self.process_batch_fn((i, K, gt, pk, weight_canvas))
                output_.append((idx, gt_hat))

        # checking batch order ... it shouldnt be needed it, but  dont wanna have nasty surprises.
        # maybe remove in the future, remove idx from the output in process_batch_fn and this is not needed.

        verified_output = []
        verified_output_pp = []
        for i in range(len(output_)):
            assert output_[i][0] == i

            output_i1 = output_[i][1][0]
            verified_output.append(output_i1)

            if self.fn_post_process_callback is not None:
                verified_output_pp.append(output_[i][1][1])
        #

        if self.fn_post_process_callback is not None:
            return np.stack(verified_output, axis=0).reshape(N, K, -1, H, W), np.stack(verified_output_pp,
                                                                                       axis=0).reshape(N, K, -1, H, W)
        else:

            return np.stack(verified_output, axis=0).reshape(N, K, -1, H, W), None  # NxKxLStepsxHxW

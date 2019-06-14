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


import contours.cutils as cutils


class LevelSetAlignmentBase:
    def __init__(self, fn_post_process_callback=None, n_workers=1, fn_debug=None, config=None):
        """

        :param fn_post_process_callback: function signature fn(evolution, pixel_wise_evol), does postprocessing inside the thread
        :param n_workers: number of worker.
        :param fn_debug: usually a function fn(image,str)... maybe a wrapper to plot.imshow(image,title)
        """
        self.fn_post_process_callback = fn_post_process_callback
        self.ignore_labels = (255,)
        self.n_workers = n_workers

        if config is None:
            self.options_dict = {
                'step_ckpts': (0, 25, 50),
                'lambda_': 0.2,
                'alpha': 1.0,
                'smoothing': 2,
                'render_radius': 2,
                'is_gt_semantic': True,
                'h_callback': cutils.compute_h_additive,
                'method': 'MLS'
            }
        else:
            self.options_dict = config
        self.fn_debug = fn_debug
        self.history = None
        print(' LevelSetAlignment config: ', self.options_dict)

    def _compute_h(self, gt_K, pK_Image, lambda_, alpha):
        if (('h_callback' in self.options_dict) == True) and (self.options_dict['h_callback'] is not None):
            _fn = self.options_dict['h_callback']
        else:
            _fn = cutils.compute_h_additive  # ...it should raise value error leaving like this to avoid breaking old experiment

        return _fn(gt_K, pK_Image, lambda_, alpha)

    def __call__(self, gt, pk):
        raise NotImplementedError()


def LevelSetAlignment(fn_post_process_callback=None, n_workers=1, fn_debug=None, config=None, method=None):
    import contours.ContourBox_MLS
    _LevelSets = {
        'MLS': contours.ContourBox_MLS.MLS,
    }

    if method is not None:
        clss_cllback = _LevelSets[method]
    elif config is None:
        clss_cllback = _LevelSets['MLS']
    else:
        clss_cllback = _LevelSets[config['method']]
    print('LevelSet Alignment n_workers: ', n_workers)
    return clss_cllback(fn_post_process_callback, n_workers, fn_debug, config)

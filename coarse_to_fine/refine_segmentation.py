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

from coarse_to_fine.input_reader import InputReader, InputReaderSemMat2, InputReaderRGBImage
import os
import tqdm
import argparse
from contours import ContourBox
import ast
from coarse_to_fine.VisualizerBox import VisualizerBox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_dir', type=str,
                        default='./sbd_reannotation')
    parser.add_argument('--coarse_dir', type=str,
                        default='')

    parser.add_argument('--in_dir', type=str,
                        default='./edge_predictions/val_images/')

    parser.add_argument('--image_dir', type=str,
                        default='./sbd/data_orig/benchmark_RELEASE/dataset')

    parser.add_argument('--val_file_list', type=str, default='')
    parser.add_argument('--n_classes', type=int, default=20)
    parser.add_argument('--n_classes_start', type=int, default=1)
    parser.add_argument('--alignment', action='store_false')
    parser.add_argument('--level_set_method', type=str, default='MLS')
    parser.add_argument('--level_set_config_dict', type=dict, default={})
    parser.add_argument('--eval_config', type=dict, default={})
    parser.add_argument('--random_pick', type=int, default=10)
    # ---
    parser.add_argument('--n_workers', type=int, default=8)
    parser.add_argument('--max_lsteps', type=int, default=10)
    parser.add_argument('--smooth_lsteps', type=int, default=1)
    parser.add_argument('--lambda_', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=100.0)
    parser.add_argument('--step_ckpts', type=str, default='')
    parser.add_argument('--middle_step', type=int, default=-1)
    parser.add_argument('--sim_coarse', action='store_true')
    parser.add_argument('--per_component_sim', action='store_true')
    parser.add_argument('--delta_coarse', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--output_dir', type=str, default='./output/refinement')
    parser.add_argument('--vis_steps', type=str, default="")
    parser.add_argument('--classes_to_keep', type=list, default=[])
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--balloon', type=float, default=0)
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--auto_show', action='store_true')
    parser.add_argument('--merge_weight', type=float, default=0)
    parser.add_argument('--per_component_click_sim', action='store_true')

    args = parser.parse_args()

    if args.alignment is True:
        level_set_config_dict = {
            'step_ckpts': [0, args.max_lsteps // 2, args.max_lsteps],  # , 25),
            'lambda_': args.lambda_,
            'alpha': args.alpha,
            'smoothing': args.smooth_lsteps,
            'render_radius': -1,
            'is_gt_semantic': True,
            'method': args.level_set_method,
            'balloon': args.balloon,
            'threshold': args.threshold,
            'merge_weight': args.merge_weight
        }

        if args.middle_step > 0:
            level_set_config_dict['step_ckpts'][1] = args.middle_step

        if args.step_ckpts != '':
            level_set_config_dict['step_ckpts'] = ast.literal_eval(args.step_ckpts)

        args.level_set_config_dict = level_set_config_dict
    if args.dataset == 'cityscapes':
        # only objects
        args.classes_to_keep = [11, 12, 13, 14, 15, 16, 17, 18]
    else:
        args.classes_to_keep = []
    return args


def prepair_contour_box(args):
    level_set_config_dict = args.level_set_config_dict
    method = args.level_set_config_dict['method']
    if method == 'MLS':
        raise ValueError()

    cbox = ContourBox.LevelSetAlignment(n_workers=1,
                                        fn_post_process_callback=None,
                                        config=level_set_config_dict)
    return cbox


def main(args):
    val_dir = args.val_dir
    in_dir = args.in_dir
    n_classes_interval = (args.n_classes_start, args.n_classes)
    vis_steps = args.vis_steps

    if vis_steps == "":
        vis_steps = []
        vis_box = None
    else:
        print('Creating VisualizerBOX')
        vis_steps = ast.literal_eval(vis_steps)
        vis_box = VisualizerBox(dataset_color='css4_fushia', plt_backend='Qt5Agg' if args.auto_show else None,
                                fig_size=(20, 10))

        vis_box.set_output_folder(os.path.join(args.output_dir, args.exp_name, 'vis'))

    if args.val_file_list == '':
        val_file_list = os.path.join(val_dir, 'val.txt')
    else:
        val_file_list = args.val_file_list

    ireader = InputReader(in_dir, val_file_list, n_classes_interval)
    ireader.set_classes_to_keep(args.classes_to_keep)

    if args.random_pick > 0:
        ireader.randompick(args.random_pick)

    # getting the reader for seg from matlab
    if args.sim_coarse is False:
        print('Using Real Coarse Data from :', args.coarse_dir)
        if not os.path.isdir(args.coarse_dir):
            raise ValueError('not dir found')
        #
        ireader_coarse_sim = InputReaderSemMat2(args.coarse_dir, val_file_list, n_classes_interval)
        ireader_coarse_sim.set_external_list(ireader._read_list)
        ireader_coarse_sim.set_classes_to_keep(args.classes_to_keep)
    else:
        raise ValueError()

    irader_semantic_init = InputReaderSemMat2(val_dir, val_file_list, n_classes_interval)
    irader_semantic_init.set_external_list(ireader._read_list)
    irader_semantic_init.set_classes_to_keep(args.classes_to_keep)

    ireader_rgb_img = InputReaderRGBImage(args.image_dir, val_file_list, n_classes_interval)
    ireader_rgb_img.set_external_list(ireader._read_list)
    ireader_rgb_img.set_classes_to_keep(args.classes_to_keep)

    # getting the reader for bdry from matlab
    cbox = prepair_contour_box(args)

    debug_output_dict = {}
    debug_ = False
    for (im_filename, pred_ch), (seg_fname, seg_coarse), (rgb_name, rgb_image), (
            seg_init_name, seg_init_ch) in tqdm.tqdm(
        zip(ireader, ireader_coarse_sim, ireader_rgb_img, irader_semantic_init), total=len(ireader)):

        assert len(pred_ch) == len(seg_coarse) == len(seg_init_ch), 'num ch should match'
        assert seg_fname == im_filename == rgb_name == seg_init_name, 'this should match'

        if len(pred_ch) == 0:
            print('skipping image: ', im_filename)
            continue

        ##checking the input images are not full resolution on cityscapes (for speed purposes)...
        w_, h_ = rgb_image.size
        if w_ == 2048:
            rgb_image = rgb_image.resize((int(w_ * 0.5), int(h_ * 0.5)))

        assert pred_ch[0].shape == seg_coarse[0].shape == tuple(reversed(rgb_image.size)) == seg_init_ch[
            0].shape, 'spatial dim should match'

        #
        output, _ = cbox({'seg': np.expand_dims(seg_coarse, 0), 'bdry': None},
                         np.expand_dims(np.stack(pred_ch), 0))

        # let's cast this, we may get a torch tensor
        output = np.array(output)
        if debug_ is True:
            debug_output_dict[im_filename] = np.copy(output)

        if vis_box is not None:
            plot_pairs = {'Fine Label': seg_init_ch,
                          'Semantic Edges': np.max(pred_ch, axis=0, keepdims=True),
                          'Real Coarse Label': seg_coarse}

            for vis_step in vis_steps:
                masks_step = output[0, :, vis_step, :, :]
                vis_step = args.level_set_config_dict['step_ckpts'][vis_step]
                plot_pairs['Refinement (Step:%i)' % vis_step] = masks_step

            vis_box.save_vis(im_filename, plot_pairs, background=rgb_image,
                             remapping_dict=ireader_coarse_sim._remapping,
                             auto_show=args.auto_show, grid=False)

    #
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    name = os.path.join(args.output_dir, 'output.npz')

    np.savez_compressed(name,
                        {'_remapping': ireader_coarse_sim._remapping})

    print('----')


if __name__ == "__main__":
    args = parse_args()
    main(args)

# STEAL
This is the official inference code for:

#### Devil Is in the Edges: Learning Semantic Boundaries from Noisy Annotations

[David Acuna](http://www.cs.toronto.edu/~davidj/), [Amlan Kar](http://www.cs.toronto.edu/~amlan/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)

CVPR 2019
**[[Paper](https://arxiv.org/abs/1904.07934)]  [[Project Page](https://nv-tlabs.github.io/STEAL/)]**

![STEAL DEMO](https://nv-tlabs.github.io/STEAL/resources/teaser_gif.gif)



## License
```
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
```
## Usage

##### Clone this repo
```bash
git clone https://github.com/nv-tlabs/STEAL
cd STEAL
 ```

#### Install dependencies

This code requires PyTorch 0.4 and python 3+. Please install dependencies by
```
pip install -r requirments.txt
```

#### Download pretrained models

Download the tar of the pretrained models from the [Google Drive Folder](https://drive.google.com/open?id=15IrPfMe9ZXJ4g0UV7tcA-LWPzCIPc1Sr), save it in 'checkpoints/', and run

```bash
cd checkpoints
tar -xvf checkpoints.tar.gz
cd ../
```


#### Inference (SBD)
```
python inference_sbd.py \
    --root_dir_val= ./data/sbd/data_aug/\
    --flist_val= ./data/sbd/data_aug/val_list.txt\
    --output_folder=./output/sbd/ \
    --ckpt=./checkpoints/sbd/model_checkpoint.pt\
```

Instructions and preprocessing scripts to download SBD and preprocess the dataset can be found here:
https://github.com/Chrisding/sbd-preprocess



#### Inference (Cityscapes)
```
python inference_cityscapes.py \
    --root_dir_val=./data/cityscapes-preprocess/data_proc \
    --flist_val=./data_proc/val.txt \
    --output_folder=./output/cityscapes/ \
    --ckpt=./checkpoints/cityscapes/model_checkpoint.pt\
```

Instructions and preprocessing scripts for Cityscapes can be found here:
https://github.com/Chrisding/cityscapes-preprocess



*Test-NMS:*
An  example of how to apply TEST-NMS using [Piotr's Structured Forest matlab toolbox](https://github.com/pdollar/edges). can be found in `utils/edges_nms.m`.
During training, we optimized for the same set of operations with r=2 (Check paper for more details)


#### Coarse-to-fine Demo
Checkout the ipython notebook that provides a simple walkthrough demonstrating how to run our model to refine coarsely annotated data.

![Coarse to Fine](https://nv-tlabs.github.io/STEAL/resources/coarse_to_fine_g.gif)

If you use this code, please cite:

```
@inproceedings{AcunaCVPR19STEAL,
title={Devil is in the Edges: Learning Semantic Boundaries from Noisy Annotations},
author={David Acuna and Amlan Kar and Sanja Fidler},
booktitle={CVPR},
year={2019}
}
```
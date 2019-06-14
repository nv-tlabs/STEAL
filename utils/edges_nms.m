%# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
%#
%# Redistribution and use in source and binary forms, with or without
%# modification, are permitted provided that the following conditions
%# are met:
%#  * Redistributions of source code must retain the above copyright
%#    notice, this list of conditions and the following disclaimer.
%#  * Redistributions in binary form must reproduce the above copyright
%#    notice, this list of conditions and the following disclaimer in the
%#    documentation and/or other materials provided with the distribution.
%#  * Neither the name of NVIDIA CORPORATION nor the names of its
%#    contributors may be used to endorse or promote products derived
%#    from this software without specific prior written permission.
%#
%# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
%# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
%# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
%# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
%# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
%# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
%# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
%# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


% This is an example of how to apply  Test-NMS using https://github.com/pdollar/edges edges toolbox
% This code was insipired by https://github.com/pdollar/edges and  https://github.com/s9xie/hed_release-deprecated/tree/master/examples/eval
% During training, we optimize for a similar procedure with r=2.

path_to_input=''
path_to_pdollar = '../edges';
path_to_pdollar_toolbox = '.../toolbox';
path_to_output = [path_to_input '_nms_22'];

addpath(genpath(path_to_pdollar));
addpath(genpath(path_to_pdollar_toolbox));
[status,msg]=mkdir(path_to_output);

iids = dir(fullfile(path_to_input, '*/*.png'));
for i = 1:length(iids)
    edge = imread(fullfile(iids(i).folder, iids(i).name));
    current_output=strrep(iids(i).folder,'/class_','_nms_22/class_');
    edge=single(edge)/255;
    edge=convTri(edge, 2);
    [Ox, Oy] = gradient2(convTri(edge, 4));
    [Oxx, ~] = gradient2(Ox);
    [Oxy, Oyy] = gradient2(Oy);
    O = mod(atan(Oyy .* sign(-(Oxy + 1e-5)) ./ (Oxx + 1e-5)), pi);
    r=2;
    edge = edgesNmsMex(edge, O, r, 5, 1.00, 6);
    [status,msg]=mkdir(current_output);
    imwrite(uint8(edge*255), fullfile(current_output, iids(i).name));
end
# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json, save_json
import json
from pathlib import Path
import re
import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--data_dir", default="/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw/Brats2021_Training_Data", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="/staging/leuven/stg_00081/jli/calibration/dataset/nnUNet_raw/default_splits_brats21.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=0, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model_final.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--TS", default=None, type=str, help="load temperature.json")
parser.add_argument("--ECE", action="store_true", help="use val_ece to calculate ECE")
parser.add_argument("--loss", default=None, type=str, help="CE, Dice, DiceCE")
parser.add_argument(
    "--pretrained_dir",
    default="./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/",
    type=str,
    help="pretrained checkpoint directory",
)


def main():
    args = parser.parse_args()
    args.test_mode = True
    # output_directory = "./outputs/" + args.exp_name
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)

    # test_loader = get_loader(args)
    test_loader, test_files = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_root = f'{args.pretrained_dir}_{args.loss}/fold_{args.fold}'
    # load ckpt
    pretrained_pth = join(fold_root, args.pretrained_model_name)# benchmark_brats21_nested_CE/fold_0/model_final.pt
    # save path
    if args.ECE:
        output_directory = join(fold_root, 'validation_ece') # val_ece
    else:
        output_directory = join(fold_root, 'test') # test
    if args.TS is not None:
        output_directory = f'{output_directory}_{args.TS}'
    os.makedirs(output_directory, exist_ok=True)

    output_directory_gt = join(output_directory,'gt')
    output_directory_seg = join(output_directory,'seg')
    output_directory_prob = join(output_directory,'prob')
    os.makedirs(output_directory_gt, exist_ok=True)
    os.makedirs(output_directory_seg, exist_ok=True)
    os.makedirs(output_directory_prob, exist_ok=True)

    model = SwinUNETR(
        img_size=128,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)
    
    ## JJ
    if args.TS is not None:
        potential_TS_path = join(fold_root, f'temperature_{args.TS}.json')  # JJ
        print(f'loading temperature from {potential_TS_path}')
        if os.path.exists(potential_TS_path):
            temperature = json.load(open(potential_TS_path, 'r'))['temperature'][0]  # in list
            print(f"Temperature Scaling: loading {temperature}")
        else:
            raise FileNotFoundError(
                f"Missing: {potential_TS_path}. Please check your model path."
            )
    else:
        temperature = None

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )
    affine = None
    with torch.no_grad():
        for i, batch in enumerate(test_loader):  # len(test_loader)
            # batch.keys(): ['fold', 'image', 'label']
            image = batch["image"].cuda()
            labels = batch["label"]# [1,3,240,240,155]
            img_name = os.path.basename(test_files[i]['label']).replace("_seg","")
            # affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            # num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
            print("Inference on case {}".format(img_name))
            logits = model_inferer_test(image).squeeze(0)
            if temperature is not None:
                logits = logits/temperature

            # -----------------softmax-------------------
            prob = F.softmax(logits, dim=0)# sum_prob = prob.sum(dim=0)
            seg = prob.detach().cpu().numpy().argmax(0).astype(np.uint8)# [3, 240, 240, 155]
            # -----------------sigmoid-------------------
            # prob = torch.sigmoid(logits)# [1, 3, 240, 240, 155]
            # seg = prob.detach().cpu().numpy().astype(np.uint8)
            # seg = (seg > 0.5).astype(np.int8)
            # -----------------seg-------------------
            # save_seg: actually overwritten here
            # seg[0]-et, seg[1]-wt, seg[2]-tc(nec)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2  # wt (green+blue+red)
            seg_out[seg[0] == 1] = 1 # enhanced (blue)
            seg_out[seg[2] == 1] = 4 # nec/core (red)
            # save_seg
            nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), join(output_directory_seg, img_name))
            # save_prob
            np.savez_compressed(join(output_directory_prob, img_name.replace('nii.gz','npz')), probabilities=prob)
            # save_gt
            nib.save(nib.Nifti1Image(labels.astype(np.uint8), affine), join(output_directory_gt, img_name))
        print("Finished inference!")


if __name__ == "__main__":
    main()

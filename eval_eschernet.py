#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import os
import einops
import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import torchvision
import json
import cv2
from skimage.io import imsave
import matplotlib.pyplot as plt

# read .exr files for RTMV dataset
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Zero123 training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/sd-image-variations-diffusers",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--T_in", type=int, default=1, help="Number of input views"
    )
    parser.add_argument(
        "--T_out", type=int, default=1, help="Number of output views"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.0,
        help="unconditional guidance scale, if guidance_scale>1.0, do_classifier_free_guidance"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help=(
            "The input data dir. Should contain the .png files (or other data files) for the task."
        ),
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="GSO25",
        help=(
            "The input data type. Chosen from GSO25, GSO3D, GSO100, RTMV, NeRF, Franka, MVDream, Text2Img"
        ),
    )
    parser.add_argument(
        "--cape_type",
        type=str,
        default="6DoF",
        help=(
            "The camera pose encoding CaPE type. Chosen from 4DoF, 6DoF"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs_eval",
        help=(
            "The output directory where the model predictions and checkpoints will be written."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers."
    )



    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images."
        )

    return args


# create angles in archimedean spiral with T_out number
import math
def get_archimedean_spiral(sphere_radius, num_steps=250):
    # x-z plane, around upper y
    '''
    https://en.wikipedia.org/wiki/Spiral, section "Spherical spiral". c = a / pi
    '''
    a = 40
    r = sphere_radius

    translations = []
    angles = []

    # i = a / 2
    i = 0.01
    while i < a:
        theta = i / a * math.pi
        x = r * math.sin(theta) * math.cos(-i)
        z = r * math.sin(-theta + math.pi) * math.sin(-i)
        y = r * - math.cos(theta)

        # translations.append((x, y, z))    # origin
        translations.append((x, z, -y))
        angles.append([np.rad2deg(-i), np.rad2deg(theta)])

        # i += a / (2 * num_steps)
        i += a / (1 * num_steps)

    return np.array(translations), np.stack(angles)

# 36 views around the circle, with elevation degree
def get_circle_traj(sphere_radius, elevation=0, num_steps=36):
    translations = []
    angles = []
    elevation = np.deg2rad(elevation)
    for i in range(num_steps):
        theta = i / num_steps * 2 * math.pi
        x = sphere_radius * math.sin(theta) * math.cos(elevation)
        z = sphere_radius * math.sin(-theta+math.pi) * math.sin(-elevation)
        y = sphere_radius * -math.cos(theta)

        translations.append((x, z, -y))
        angles.append([np.rad2deg(-elevation), np.rad2deg(theta)])

    return np.array(translations), np.stack(angles)



def look_at(origin, target, up):
    forward = (target - origin)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    new_up = np.cross(forward, right)
    rotation_matrix = np.column_stack((right, new_up, -forward, target))
    matrix = np.row_stack((rotation_matrix, [0, 0, 0, 1]))
    return matrix

# from carvekit.api.high import HiInterface
# def create_carvekit_interface():
#     # Check doc strings for more information
#     interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
#                             batch_size_seg=5,
#                             batch_size_matting=1,
#                             device='cuda' if torch.cuda.is_available() else 'cpu',
#                             seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
#                             matting_mask_size=2048,
#                             trimap_prob_threshold=231,
#                             trimap_dilation=30,
#                             trimap_erosion_iters=5,
#                             fp16=False)
#
#     return interface

import rembg
def create_rembg_interface():
    rembg_session = rembg.new_session()

    return rembg_session

def main(args):
    if args.seed is not None:
        set_seed(args.seed)

    CaPE_TYPE = args.cape_type
    if CaPE_TYPE == "6DoF":
        import sys
        sys.path.insert(0, "./6DoF/")
        # use the customized diffusers modules
        from diffusers import DDIMScheduler
        from dataset import get_pose
        from CN_encoder import CN_encoder
        from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline

    elif CaPE_TYPE == "4DoF":
        import sys
        sys.path.insert(0, "./4DoF/")
        # use the customized diffusers modules
        from diffusers import DDIMScheduler
        from dataset import get_pose
        from CN_encoder import CN_encoder
        from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
    else:
        raise ValueError("CaPE_TYPE must be chosen from 4DoF, 6DoF")

    # from dataset import get_pose
    # from CN_encoder import CN_encoder
    # from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline

    DATA_DIR = args.data_dir
    DATA_TYPE = args.data_type

    if DATA_TYPE == "GSO25":
        T_in_DATA_TYPE = "render_mvs_25" # same condition for GSO
        T_out_DATA_TYPE = "render_mvs_25"   # for 2D metrics
        T_out = 25
    elif DATA_TYPE == "GSO25_6dof":
        T_in_DATA_TYPE = "render_6dof_25" # same condition for GSO
        T_out_DATA_TYPE = "render_6dof_25"   # for 2D metrics
        T_out = 25
    elif DATA_TYPE == "GSO3D":
        T_in_DATA_TYPE = "render_mvs_25"  # same condition for GSO
        T_out_DATA_TYPE = "render_sync_36_single" # for 3D metrics
        T_out = 36
    elif DATA_TYPE == "GSO100":
        T_in_DATA_TYPE = "render_mvs_25"  # same condition for GSO
        T_out_DATA_TYPE = "render_spiral_100"   # for 360 gif
        T_out = 100
    elif DATA_TYPE == "NeRF":
        T_out = 200
    elif DATA_TYPE == "RTMV":
        T_out = 20
    elif DATA_TYPE == "Franka":
        T_out = 100 # do a 360 gif
    elif DATA_TYPE == "MVDream":
        T_out = 100 # do a 360 gif
    elif DATA_TYPE == "Text2Img":
        T_out = 100  # do a 360 gif
    elif DATA_TYPE == "dust3r":
        # carvekit = create_carvekit_interface()
        rembg_session = create_rembg_interface()
        T_out = 50  # do a 360 gif
        # get the number of .png files in the folder
        obj_names = [f for f in os.listdir(DATA_DIR+"/user_object") if f.endswith('.png')]
        args.T_in = len(obj_names)
    else:
        raise NotImplementedError

    T_in = args.T_in
    OUTPUT_DIR= f"logs_{CaPE_TYPE}/{DATA_TYPE}/N{T_in}M{T_out}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # get all folders in DATA_DIR
    if DATA_TYPE == "Text2Img":
        # get all rgba_png in DATA_DIR
        obj_names = [f for f in os.listdir(DATA_DIR) if f.endswith('rgba.png')]
    else:
        obj_names = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

    weight_dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = args.resolution, args.resolution
    bg_color = [1., 1., 1., 1.]
    radius = 2.2 #1.5 #1.8    # Objaverse training radius [1.5, 2.2]
    # radius_4dof = np.pi * (np.log(radius) - np.log(1.5)) / (np.log(2.2)-np.log(1.5))

    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution)),  # 256, 256
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    # Init pipeline
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                              revision=args.revision)
    image_encoder = CN_encoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        scheduler=scheduler,
        image_encoder=None,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=weight_dtype,
    )
    pipeline.image_encoder = image_encoder
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
        # enable vae slicing
        pipeline.enable_vae_slicing()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)


    for obj_name in tqdm(obj_names):
        print(f"Processing {obj_name}")
        if DATA_TYPE == "NeRF":
            if os.path.exists(os.path.join(args.output_dir, obj_name, "output.gif")):
                continue
            # load train info
            with open(os.path.join(DATA_DIR, obj_name, "transforms_train.json"), "r") as f:
                train_info = json.load(f)["frames"]
            # load test info
            with open(os.path.join(DATA_DIR, obj_name, "transforms_test.json"), "r") as f:
                test_info = json.load(f)["frames"]

            # find the radius [min_t, max_t] of the object, we later scale it to training radius [1.5, 2.2]
            max_t = 0
            min_t = 100
            for i in range(len(train_info)):
                pose = np.array(train_info[i]["transform_matrix"]).reshape(4, 4)
                translation = pose[:3, -1]
                radii = np.linalg.norm(translation)
                if max_t < radii:
                    max_t = radii
                if min_t > radii:
                    min_t = radii
            info_dir = os.path.join("metrics/NeRF_idx", obj_name)
            assert os.path.exists(info_dir)  # use fixed train index
            train_index = np.load(os.path.join(info_dir, f"train_N{T_in}M20_random.npy"))
            test_index = np.arange(len(test_info))  # use all test views
        elif DATA_TYPE == "Franka":
            angles_in = np.load(os.path.join(DATA_DIR, obj_name, "angles.npy"))  # azimuth, elevation in radians
            assert T_in <= len(angles_in)
            total_index = np.arange(0, len(angles_in))  # num of input views
            # random shuffle total_index
            np.random.shuffle(total_index)
            train_index = total_index[:T_in]
            xyzs, angles_out = get_archimedean_spiral(radius, T_out)
            origin = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            test_index = np.arange(len(angles_out))  # use all 100 test views
        elif DATA_TYPE == "MVDream":    # 4 input views front right back left
            angles_in = []
            for polar in [90]:  # 1
                for azimu in np.arange(0, 360, 90):  # 4
                    angles_in.append(np.array([azimu, polar]))
            assert T_in == len(angles_in)
            xyzs, angles_out = get_archimedean_spiral(radius, T_out)
            origin = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            train_index = np.arange(T_in)
            test_index = np.arange(T_out)
        elif DATA_TYPE == "Text2Img":    # 1 input view
            angles_in = []
            angles_in.append(np.array([0, 90]))
            assert T_in == len(angles_in)
            xyzs, angles_out = get_archimedean_spiral(radius, T_out)
            origin = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            train_index = np.arange(T_in)
            test_index = np.arange(T_out)
        elif DATA_TYPE == "dust3r":
            # TODO full archimedean spiral traj
            # xyzs, angles_out = get_archimedean_spiral(radius, T_out)
            # TODO only top circle traj
            xyzs, angles_out = get_archimedean_spiral(1.5, 100)
            xyzs = xyzs[:T_out]
            angles_out = angles_out[:T_out]
            # # TODO circle traj
            # xyzs, angles_out = get_circle_traj(radius, elevation=30, num_steps=T_out)
            origin = np.array([0, 0, 0])
            up = np.array([0, 0, 1])
            train_index = np.arange(T_in)
            test_index = np.arange(T_out)
            # get the max_t
            radii = np.load(os.path.join(DATA_DIR, obj_name, "radii.npy"))
            max_t = np.max(radii)
            min_t = np.min(radii)
        else:
            train_index = np.arange(T_in)
            test_index = np.arange(T_out)


        # prepare input img + pose, output pose
        input_image = []
        pose_in = []
        pose_out = []
        gt_image = []
        for T_in_index in train_index:
            if DATA_TYPE == "RTMV":
                img_path = os.path.join(DATA_DIR, obj_name, '%05d.exr' % T_in_index)
                input_im = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                img = cv2.cvtColor(input_im, cv2.COLOR_BGR2RGB, input_im)
                img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
                input_image.append(image_transforms(img))
                # load input pose
                pose_path = os.path.join(DATA_DIR, obj_name, '%05d.json' % T_in_index)
                with open(pose_path, "r") as f:
                    pose_dict = json.load(f)
                input_RT = np.array(pose_dict["camera_data"]["cam2world"]).T
                input_RT = np.linalg.inv(input_RT)[:3]
                pose_in.append(get_pose(np.concatenate([input_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
            else:
                if DATA_TYPE == "NeRF":
                    img_path = os.path.join(DATA_DIR, obj_name, train_info[T_in_index]["file_path"] + ".png")
                    pose = np.array(train_info[T_in_index]["transform_matrix"])
                    if CaPE_TYPE == "6DoF":
                        # blender to opencv
                        pose[1:3, :] *= -1
                        pose = np.linalg.inv(pose)
                        # scale radius to [1.5, 2.2]
                        pose[:3, 3] *= 1. / max_t * radius
                    elif CaPE_TYPE == "4DoF":
                        pose = np.linalg.inv(pose)
                    pose_in.append(torch.from_numpy(get_pose(pose)))
                elif DATA_TYPE == "Franka":
                    img_path = os.path.join(DATA_DIR, obj_name, "images_rgba", f"frame{T_in_index:06d}.png")
                    azimuth, elevation = np.rad2deg(angles_in[T_in_index])
                    print("input angles index", T_in_index, "azimuth", azimuth, "elevation", elevation)
                    if CaPE_TYPE == "4DoF":
                        pose_in.append(torch.from_numpy([np.deg2rad(90. - elevation), np.deg2rad(azimuth - 180), 0., 0.]))
                    elif CaPE_TYPE == "6DoF":
                        neg_i = np.deg2rad(azimuth - 180)
                        neg_theta = np.deg2rad(90. - elevation)
                        xyz = np.array([np.sin(neg_theta) * np.cos(neg_i),
                                        np.sin(-neg_theta + np.pi) * np.sin(neg_i),
                                        np.cos(neg_theta)]) * radius
                        pose = look_at(origin, xyz, up)
                        pose = np.linalg.inv(pose)
                        pose[2, :] *= -1
                        pose_in.append(torch.from_numpy(get_pose(pose)))
                elif DATA_TYPE == "MVDream" or DATA_TYPE == "Text2Img":
                    if DATA_TYPE == "MVDream":
                        img_path = os.path.join(DATA_DIR, obj_name, f"{T_in_index}_rgba.png")
                    elif DATA_TYPE == "Text2Img":
                        img_path = os.path.join(DATA_DIR, obj_name)
                    azimuth, polar = angles_in[T_in_index]
                    if CaPE_TYPE == "4DoF":
                        pose_in.append(torch.tensor([np.deg2rad(polar), np.deg2rad(azimuth), 0., 0.]))
                    elif CaPE_TYPE == "6DoF":
                        neg_theta = np.deg2rad(polar)
                        neg_i = np.deg2rad(azimuth)
                        xyz = np.array([np.sin(neg_theta) * np.cos(neg_i),
                                        np.sin(-neg_theta + np.pi) * np.sin(neg_i),
                                        np.cos(neg_theta)]) * radius
                        pose = look_at(origin, xyz, up)
                        pose = np.linalg.inv(pose)
                        pose[2, :] *= -1
                        pose_in.append(torch.from_numpy(get_pose(pose)))
                elif DATA_TYPE == "dust3r": # TODO get the object coordinate, now one of the camera is the center
                    img_path = os.path.join(DATA_DIR, obj_name, "%03d.png" % T_in_index)
                    pose = get_pose(np.linalg.inv(np.load(os.path.join(DATA_DIR, obj_name, "%03d.npy" % T_in_index))))
                    pose[1:3, :] *= -1
                    # scale radius to [1.5, 2.2]
                    pose[:3, 3] *= 1. / max_t * radius
                    pose_in.append(torch.from_numpy(pose))
                else:   # GSO
                    img_path = os.path.join(DATA_DIR, obj_name, T_in_DATA_TYPE, "model/%03d.png" % T_in_index)
                    pose_path = os.path.join(DATA_DIR, obj_name, T_in_DATA_TYPE, "model/%03d.npy" % T_in_index)
                    if T_in_DATA_TYPE == "render_mvs_25" or T_in_DATA_TYPE == "render_6dof_25": # blender coordinate
                        pose_in.append(get_pose(np.concatenate([np.load(pose_path)[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
                    else:   # opencv coordinate
                        pose = get_pose(np.concatenate([np.load(pose_path)[:3, :], np.array([[0, 0, 0, 1]])], axis=0))
                        pose[1:3, :] *= -1  # pose out 36 is in opencv coordinate, pose in 25 is in blender coordinate
                        pose_in.append(torch.from_numpy(pose))
                    # pose_in.append(get_pose(np.concatenate([np.load(pose_path)[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))

                # load image
                img = plt.imread(img_path)
                if (img.shape[-1] == 3 or (img[:,:,-1] == 1).all()) and DATA_TYPE == "dust3r":
                    img_pil = Image.fromarray(np.uint8(img * 255.)).convert("RGB")   # to PIL image

                    ## use carvekit
                    # image_without_background = carvekit([img_pil])[0]
                    # image_without_background = np.array(image_without_background)
                    # est_seg = image_without_background > 127
                    # foreground = est_seg[:, :, -1].astype(np.bool_)
                    # img = np.concatenate([img[:,:,:3], foreground[:, :, np.newaxis]], axis=-1)

                    # use rembg
                    image = rembg.remove(img_pil, session=rembg_session)
                    foreground = np.array(image)[:,:,-1] > 127
                    img = np.concatenate([img[:,:,:3], foreground[:, :, np.newaxis]], axis=-1)


                img[img[:, :, -1] == 0.] = bg_color
                img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
                input_image.append(image_transforms(img))


        for T_out_index in test_index:
            if DATA_TYPE == "RTMV":
                img_path = os.path.join(DATA_DIR, obj_name, '%05d.exr' % T_out_index)
                gt_im = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                img = cv2.cvtColor(gt_im, cv2.COLOR_BGR2RGB, gt_im)
                img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
                gt_image.append(image_transforms(img))
                # load pose
                pose_path = os.path.join(DATA_DIR, obj_name, '%05d.json' % T_out_index)
                with open(pose_path, "r") as f:
                    pose_dict = json.load(f)
                output_RT = np.array(pose_dict["camera_data"]["cam2world"]).T
                output_RT = np.linalg.inv(output_RT)[:3]
                pose_out.append(get_pose(np.concatenate([output_RT[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
            else:
                if DATA_TYPE == "NeRF":
                    img_path = os.path.join(DATA_DIR, obj_name, test_info[T_out_index]["file_path"] + ".png")
                    pose = np.array(test_info[T_out_index]["transform_matrix"])
                    if CaPE_TYPE == "6DoF":
                        # blender to opencv
                        pose[1:3, :] *= -1
                        pose = np.linalg.inv(pose)
                        # scale radius to [1.5, 2.2]
                        pose[:3, 3] *= 1. / max_t * radius
                    elif CaPE_TYPE == "4DoF":
                        pose = np.linalg.inv(pose)
                    pose_out.append(torch.from_numpy(get_pose(pose)))
                elif DATA_TYPE == "Franka":
                    img_path = None
                    azimuth, polar = angles_out[T_out_index]
                    if CaPE_TYPE == "4DoF":
                        pose_out.append(torch.from_numpy([np.deg2rad(polar), np.deg2rad(azimuth), 0., 0.]))
                    elif CaPE_TYPE == "6DoF":
                        pose = look_at(origin, xyzs[T_out_index], up)
                        neg_theta = np.deg2rad(polar)
                        neg_i = np.deg2rad(azimuth)
                        xyz = np.array([np.sin(neg_theta) * np.cos(neg_i),
                                        np.sin(-neg_theta + np.pi) * np.sin(neg_i),
                                        np.cos(neg_theta)]) * radius
                        assert np.allclose(xyzs[T_out_index], xyz)
                        pose = np.linalg.inv(pose)
                        pose[2, :] *= -1
                        pose_out.append(torch.from_numpy(get_pose(pose)))
                elif DATA_TYPE == "MVDream" or DATA_TYPE == "Text2Img" or DATA_TYPE == "dust3r":
                    img_path = None
                    azimuth, polar = angles_out[T_out_index]
                    if CaPE_TYPE == "4DoF":
                        pose_out.append(torch.tensor([np.deg2rad(polar), np.deg2rad(azimuth), 0., 0.]))
                    elif CaPE_TYPE == "6DoF":
                        pose = look_at(origin, xyzs[T_out_index], up)
                        pose = np.linalg.inv(pose)
                        pose[2, :] *= -1
                        pose_out.append(torch.from_numpy(get_pose(pose)))
                else:   # GSO
                    img_path = os.path.join(DATA_DIR, obj_name, T_out_DATA_TYPE, "model/%03d.png" % T_out_index)
                    pose_path = os.path.join(DATA_DIR, obj_name, T_out_DATA_TYPE, "model/%03d.npy" % T_out_index)
                    if T_out_DATA_TYPE == "render_mvs_25" or T_out_DATA_TYPE == "render_6dof_25": # blender coordinate
                        pose_out.append(get_pose(np.concatenate([np.load(pose_path)[:3, :], np.array([[0, 0, 0, 1]])], axis=0)))
                    else:   # opencv coordinate
                        pose = get_pose(np.concatenate([np.load(pose_path)[:3, :], np.array([[0, 0, 0, 1]])], axis=0))
                        pose[1:3, :] *= -1  # pose out 36 is in opencv coordinate, pose in 25 is in blender coordinate
                        pose_out.append(torch.from_numpy(pose))

                # load image
                if img_path is not None:    # sometimes don't have GT target view image
                    img = plt.imread(img_path)
                    img[img[:, :, -1] == 0.] = bg_color
                    img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
                    gt_image.append(image_transforms(img))

        # [B, T, C, H, W]
        input_image = torch.stack(input_image, dim=0).to(device).to(weight_dtype).unsqueeze(0)
        if len(gt_image)>0:
            gt_image = torch.stack(gt_image, dim=0).to(device).to(weight_dtype).unsqueeze(0)
        # [B, T, 4]
        pose_in = np.stack(pose_in)
        pose_out = np.stack(pose_out)

        if CaPE_TYPE == "6DoF":
            pose_in_inv = np.linalg.inv(pose_in).transpose([0, 2, 1])
            pose_out_inv = np.linalg.inv(pose_out).transpose([0, 2, 1])
            pose_in_inv = torch.from_numpy(pose_in_inv).to(device).to(weight_dtype).unsqueeze(0)
            pose_out_inv = torch.from_numpy(pose_out_inv).to(device).to(weight_dtype).unsqueeze(0)


        pose_in = torch.from_numpy(pose_in).to(device).to(weight_dtype).unsqueeze(0)
        pose_out = torch.from_numpy(pose_out).to(device).to(weight_dtype).unsqueeze(0)

        input_image = einops.rearrange(input_image, "b t c h w -> (b t) c h w")
        if len(gt_image)>0:
            gt_image = einops.rearrange(gt_image, "b t c h w -> (b t) c h w")
        assert T_in == input_image.shape[0]
        assert T_in == pose_in.shape[1]
        assert T_out == pose_out.shape[1]

        # run inference
        if CaPE_TYPE == "6DoF":
            with torch.autocast("cuda"):
                image = pipeline(input_imgs=input_image, prompt_imgs=input_image, poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
                                 height=h, width=w, T_in=T_in, T_out=T_out,
                                 guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generator,
                                 output_type="numpy").images
        elif CaPE_TYPE == "4DoF":
            with torch.autocast("cuda"):
                image = pipeline(input_imgs=input_image, prompt_imgs=input_image, poses=[pose_out, pose_in],
                                 height=h, width=w, T_in=T_in, T_out=T_out,
                                 guidance_scale=args.guidance_scale, num_inference_steps=50, generator=generator,
                                 output_type="numpy").images

        # save results
        output_dir = os.path.join(OUTPUT_DIR, obj_name)
        os.makedirs(output_dir, exist_ok=True)
        # save input image for visualization
        imsave(os.path.join(output_dir, 'input.png'),
               ((np.concatenate(input_image.permute(0, 2, 3, 1).cpu().numpy(), 1) + 1) / 2 * 255).astype(np.uint8))
        # save output image
        if T_out >= 30:
            # save to N imgs
            for i in range(T_out):
                imsave(os.path.join(output_dir, f'{i}.png'), (image[i] * 255).astype(np.uint8))
            # make a gif
            frames = [Image.fromarray((image[i] * 255).astype(np.uint8)) for i in range(T_out)]
            frame_one = frames[0]
            frame_one.save(os.path.join(output_dir, "output.gif"), format="GIF", append_images=frames,
                           save_all=True, duration=50, loop=1)
        else:
            imsave(os.path.join(output_dir, '0.png'), (np.concatenate(image, 1) * 255).astype(np.uint8))
            # save gt for visualization
            if len(gt_image)>0:
                imsave(os.path.join(output_dir, 'gt.png'),
                       ((np.concatenate(gt_image.permute(0, 2, 3, 1).cpu().numpy(), 1) + 1) / 2 * 255).astype(np.uint8))




if __name__ == "__main__":
    args = parse_args()
    main(args)

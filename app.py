import spaces
import torch
print("cuda is available: ", torch.cuda.is_available())

import gradio as gr
import os
import shutil
import rembg
import numpy as np
import math
import open3d as o3d
from PIL import Image
import torchvision
import trimesh
from skimage.io import imsave
import imageio
import cv2
import matplotlib.pyplot as pl
pl.ion()

CaPE_TYPE = "6DoF"
device = 'cuda' #if torch.cuda.is_available() else 'cpu'
weight_dtype = torch.float16
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# EscherNet
# create angles in archimedean spiral with N steps
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

def look_at(origin, target, up):
    forward = (target - origin)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    new_up = np.cross(forward, right)
    rotation_matrix = np.column_stack((right, new_up, -forward, target))
    matrix = np.row_stack((rotation_matrix, [0, 0, 0, 1]))
    return matrix

import einops
import sys

sys.path.insert(0, "./6DoF/")   # TODO change it when deploying
# use the customized diffusers modules
from diffusers import DDIMScheduler
from dataset import get_pose
from CN_encoder import CN_encoder
from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
from segment_anything import sam_model_registry, SamPredictor
import rembg

pretrained_model_name_or_path = "kxic/EscherNet_demo"
resolution = 256
h,w = resolution,resolution
guidance_scale = 3.0
radius = 2.2
bg_color = [1., 1., 1., 1.]
image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((resolution, resolution)),  # 256, 256
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )
xyzs_spiral, angles_spiral = get_archimedean_spiral(1.5, 200)
# only half toop
xyzs_spiral = xyzs_spiral[:100]
angles_spiral = angles_spiral[:100]

# Init pipeline
scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", revision=None)
image_encoder = CN_encoder.from_pretrained(pretrained_model_name_or_path, subfolder="image_encoder", revision=None)
pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    revision=None,
    scheduler=scheduler,
    image_encoder=None,
    safety_checker=None,
    feature_extractor=None,
    torch_dtype=weight_dtype,
)
pipeline.image_encoder = image_encoder.to(weight_dtype)

pipeline.set_progress_bar_config(disable=False)

pipeline = pipeline.to(device)

# pipeline.enable_xformers_memory_efficient_attention()
# enable vae slicing
pipeline.enable_vae_slicing()
# pipeline.enable_xformers_memory_efficient_attention()


#### object segmentation
def sam_init():
    sam_checkpoint = os.path.join("./sam_pt/sam_vit_h_4b8939.pth")
    if os.path.exists(sam_checkpoint) is False:
        os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ./sam_pt/")
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)
    return predictor

rembg_session = rembg.new_session()
predictor = sam_init()



@spaces.GPU(duration=120)
def run_eschernet(eschernet_input_dict, sample_steps, sample_seed, nvs_num, nvs_mode):
    # set the random seed
    generator = torch.Generator(device=device).manual_seed(sample_seed)
    # generator = None
    T_out = nvs_num
    T_in = len(eschernet_input_dict['imgs'])
    ####### output pose
    # TODO choose T_out number of poses sequentially from the spiral
    xyzs = xyzs_spiral[::(len(xyzs_spiral) // T_out)]
    angles_out = angles_spiral[::(len(xyzs_spiral) // T_out)]

    ####### input's max radius for translation scaling
    radii = eschernet_input_dict['radii']
    max_t = np.max(radii)
    min_t = np.min(radii)

    ####### input pose
    pose_in = []
    for T_in_index in range(T_in):
        pose = get_pose(np.linalg.inv(eschernet_input_dict['poses'][T_in_index]))
        pose[1:3, :] *= -1   # coordinate system conversion
        pose[3, 3] *= 1. / max_t * radius    # scale radius to [1.5, 2.2]
        pose_in.append(torch.from_numpy(pose))

    ####### input image
    img = eschernet_input_dict['imgs'] / 255.
    img[img[:, :, :, -1] == 0.] = bg_color
    # TODO batch image_transforms
    input_image = [image_transforms(Image.fromarray(np.uint8(im[:, :, :3] * 255.)).convert("RGB")) for im in img]

    ####### nvs pose
    pose_out = []
    for T_out_index in range(T_out):
        azimuth, polar = angles_out[T_out_index]
        if CaPE_TYPE == "4DoF":
            pose_out.append(torch.tensor([np.deg2rad(polar), np.deg2rad(azimuth), 0., 0.]))
        elif CaPE_TYPE == "6DoF":
            pose = look_at(origin=np.array([0, 0, 0]), target=xyzs[T_out_index], up=np.array([0, 0, 1]))
            pose = np.linalg.inv(pose)
            pose[2, :] *= -1
            pose_out.append(torch.from_numpy(get_pose(pose)))



    # [B, T, C, H, W]
    input_image = torch.stack(input_image, dim=0).to(device).to(weight_dtype).unsqueeze(0)
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
    assert T_in == input_image.shape[0]
    assert T_in == pose_in.shape[1]
    assert T_out == pose_out.shape[1]

    # run inference
    # pipeline.to(device)
    pipeline.enable_xformers_memory_efficient_attention()
    image = pipeline(input_imgs=input_image, prompt_imgs=input_image,
                         poses=[[pose_out, pose_out_inv], [pose_in, pose_in_inv]],
                         height=h, width=w, T_in=T_in, T_out=T_out,
                         guidance_scale=guidance_scale, num_inference_steps=50, generator=generator,
                         output_type="numpy").images

    # save output image
    output_dir = os.path.join(tmpdirname, "eschernet")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # # save to N imgs
    # for i in range(T_out):
    #     imsave(os.path.join(output_dir, f'{i}.png'), (image[i] * 255).astype(np.uint8))
    # make a gif
    frames = [Image.fromarray((image[i] * 255).astype(np.uint8)) for i in range(T_out)]
    # frame_one = frames[0]
    # frame_one.save(os.path.join(output_dir, "output.gif"), format="GIF", append_images=frames,
    #                save_all=True, duration=50, loop=1)

    # get a video
    video_path = os.path.join(output_dir, "output.mp4")
    imageio.mimwrite(video_path, np.stack(frames), fps=10, codec='h264')


    return video_path

# TODO mesh it
@spaces.GPU(duration=120)
def make3d():
    pass



############################ Dust3r as Pose Estimation ############################
from scipy.spatial.transform import Rotation
import copy

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import math

@spaces.GPU(duration=120)
def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, same_focals=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world)
    if not same_focals:
        assert (len(cams2world) == len(focals))
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # add axes
    scene.add_geometry(trimesh.creation.axis(axis_length=0.5, axis_radius=0.001))

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        if same_focals:
            focal = focals[0]
        else:
            focal = focals[i]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focal,
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

@spaces.GPU(duration=120)
def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, same_focals=False):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = to_numpy(scene.imgs)
    focals = to_numpy(scene.get_focals().cpu())
    # cams2world = to_numpy(scene.get_im_poses().cpu())
    # TODO use the vis_poses
    cams2world = scene.vis_poses

    # 3D pointcloud from depthmap, poses and intrinsics
    # pts3d = to_numpy(scene.get_pts3d())
    # TODO use the vis_poses
    pts3d = scene.vis_pts3d
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent,
                                        same_focals=same_focals)

@spaces.GPU(duration=120)
def get_reconstructed_scene(filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid, same_focals):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    silent = False
    image_size = 224
    # remove the directory if it already exists
    outdir = tmpdirname
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    imgs, imgs_rgba = load_images(filelist, size=image_size, verbose=not silent, do_remove_background=True, rembg_session=rembg_session, predictor=predictor)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent, same_focals=same_focals)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    # outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
    #                                   clean_depth, transparent_cams, cam_size, same_focals=same_focals)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    # depths = to_numpy(scene.get_depthmaps())
    # confs = to_numpy([c for c in scene.im_conf])
    # cmap = pl.get_cmap('jet')
    # depths_max = max([d.max() for d in depths])
    # depths = [d / depths_max for d in depths]
    # confs_max = max([d.max() for d in confs])
    # confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    rgbaimg = []
    for i in range(len(rgbimg)):   # when only 1 image, scene.imgs is two
        imgs.append(rgbimg[i])
        # imgs.append(rgb(depths[i]))
        # imgs.append(rgb(confs[i]))
        # imgs.append(imgs_rgba[i])
        if len(imgs_rgba) == 1 and i == 1:
            imgs.append(imgs_rgba[0])
            rgbaimg.append(np.array(imgs_rgba[0]))
        else:
            imgs.append(imgs_rgba[i])
            rgbaimg.append(np.array(imgs_rgba[i]))

    rgbaimg = np.array(rgbaimg)

    # for eschernet
    # get optimized values from scene
    rgbimg = to_numpy(scene.imgs)
    # focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())

    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    obj_mask = rgbaimg[..., 3] > 0

    # TODO set global coordinate system at the center of the scene, z-axis is up
    pts = np.concatenate([p[m] for p, m in zip(pts3d, msk)]).reshape(-1, 3)
    pts_obj = np.concatenate([p[m&obj_m] for p, m, obj_m in zip(pts3d, msk, obj_mask)]).reshape(-1, 3)
    centroid = np.mean(pts_obj, axis=0) # obj center
    obj2world = np.eye(4)
    obj2world[:3, 3] = -centroid  # T_wc

    # get z_up vector
    # TODO fit a plane and get the normal vector
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    # get the normalised normal vector dim = 3
    normal = plane_model[:3] / np.linalg.norm(plane_model[:3])
    # the normal direction should be pointing up
    if normal[1] < 0:
        normal = -normal
    # print("normal", normal)

    # # TODO z-up 180
    # z_up = np.array([[1,0,0,0],
    #                       [0,-1,0,0],
    #                       [0,0,-1,0],
    #                       [0,0,0,1]])
    # obj2world = z_up @ obj2world

    # # avg the y
    # z_up_avg = cams2world[:,:3,3].sum(0) / np.linalg.norm(cams2world[:,:3,3].sum(0), axis=-1)    # average direction in cam coordinate
    # # import pdb; pdb.set_trace()
    # rot_axis = np.cross(np.array([0, 0, 1]), z_up_avg)
    # rot_angle = np.arccos(np.dot(np.array([0, 0, 1]), z_up_avg) / (np.linalg.norm(z_up_avg) + 1e-6))
    # rot = Rotation.from_rotvec(rot_angle * rot_axis)
    # z_up = np.eye(4)
    # z_up[:3, :3] = rot.as_matrix()

    # get the rotation matrix from normal to z-axis
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(normal, z_axis)
    rot_angle = np.arccos(np.dot(normal, z_axis) / (np.linalg.norm(normal) + 1e-6))
    rot = Rotation.from_rotvec(rot_angle * rot_axis)
    z_up = np.eye(4)
    z_up[:3, :3] = rot.as_matrix()
    obj2world = z_up @ obj2world
    # flip 180
    flip_rot = np.array([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])
    obj2world = flip_rot @ obj2world

    # get new cams2obj
    cams2obj = []
    for i, cam2world in enumerate(cams2world):
        cams2obj.append(obj2world @ cam2world)
    # TODO transform pts3d to the new coordinate system
    for i, pts in enumerate(pts3d):
        pts3d[i] = (obj2world @ np.concatenate([pts, np.ones_like(pts)[..., :1]], axis=-1).transpose(2, 0, 1).reshape(4,
                                                                                                                      -1)) \
                       .reshape(4, pts.shape[0], pts.shape[1]).transpose(1, 2, 0)[..., :3]
    cams2world = np.array(cams2obj)
    # TODO rewrite hack
    scene.vis_poses = cams2world.copy()
    scene.vis_pts3d = pts3d.copy()

    # TODO save cams2world and rgbimg to each file, file name "000.npy", "001.npy", ... and "000.png", "001.png", ...
    for i, (img, img_rgba, pose) in enumerate(zip(rgbimg, rgbaimg, cams2world)):
        np.save(os.path.join(outdir, f"{i:03d}.npy"), pose)
        pl.imsave(os.path.join(outdir, f"{i:03d}.png"), img)
        pl.imsave(os.path.join(outdir, f"{i:03d}_rgba.png"), img_rgba)
        # np.save(os.path.join(outdir, f"{i:03d}_focal.npy"), to_numpy(focal))
    # save the min/max radius of camera
    radii = np.linalg.norm(np.linalg.inv(cams2world)[..., :3, 3])
    np.save(os.path.join(outdir, "radii.npy"), radii)

    eschernet_input = {"poses": cams2world,
                       "radii": radii,
                       "imgs": rgbaimg}
    print("got eschernet input")
    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, same_focals=same_focals)

    return scene, outfile, imgs, eschernet_input


def set_scenegraph_options(inputfiles, winsize, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize = max(1, math.ceil((num_files - 1) / 2))
    if scenegraph_type == "swin":
        winsize = gr.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=True)
        refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    elif scenegraph_type == "oneref":
        winsize = gr.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=True)
    else:
        winsize = gr.Slider(label="Scene Graph: Window Size", value=max_winsize,
                                minimum=1, maximum=max_winsize, step=1, visible=False)
        refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0,
                              maximum=num_files - 1, step=1, visible=False)
    return winsize, refid


def get_examples(path):
    objs = []
    for obj_name in sorted(os.listdir(path)):
        img_files = []
        for img_file in sorted(os.listdir(os.path.join(path, obj_name))):
            img_files.append(os.path.join(path, obj_name, img_file))
        objs.append([img_files])
    print("objs = ", objs)
    return objs

def preview_input(inputfiles):
    if inputfiles is None:
        return None
    imgs = []
    for img_file in inputfiles:
        img = pl.imread(img_file)
        imgs.append(img)
    return imgs

# def main():
# dustr init
silent = False
image_size = 224
weights_path = 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth'
model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to(device)
# dust3r will write the 3D model inside tmpdirname
# with tempfile.TemporaryDirectory(suffix='dust3r_gradio_demo') as tmpdirname:
tmpdirname = os.path.join('logs/user_object')
# remove the directory if it already exists
if os.path.exists(tmpdirname):
    shutil.rmtree(tmpdirname)
os.makedirs(tmpdirname, exist_ok=True)
if not silent:
    print('Outputing stuff in', tmpdirname)

_HEADER_ = '''
<h2><b>[CVPR'24 Oral] EscherNet: A Generative Model for Scalable View Synthesis</b></h2>
<b>EscherNet</b> is a multiview diffusion model for scalable generative any-to-any number/pose novel view synthesis. 

Image views are treated as tokens and the camera pose is encoded by <b>CaPE (Camera Positional Encoding)</b>.

<a href='https://kxhit.github.io/EscherNet' target='_blank'>Project</a> <b>|</b>
<a href='https://github.com/kxhit/EscherNet' target='_blank'>GitHub</a> <b>|</b>
<a href='https://arxiv.org/abs/2402.03908' target='_blank'>ArXiv</a>

<h4><b>Tips:</b></h4>

- Our model can take <b>any number input images</b>. The more images you provide (>=3 for this demo), the better the results.

- Our model can generate <b>any number and any pose</b> novel views. You can specify the number of views you want to generate. In this demo, we set novel views on an <b>archemedian spiral</b> for simplicity.

- The pose estimation is done using <a href='https://github.com/naver/dust3r' target='_blank'>DUSt3R</a>. You can also provide your own poses or get pose via any SLAM system.

- The current checkpoint supports 6DoF camera pose and is trained on 30k 3D <a href='https://objaverse.allenai.org/' target='_blank'>Objaverse</a> objects for demo. Scaling is on the roadmap!

'''

_CITE_ = r"""
📝 <b>Citation</b>:
```bibtex
@article{kong2024eschernet,
    title={EscherNet: A Generative Model for Scalable View Synthesis},
    author={Kong, Xin and Liu, Shikun and Lyu, Xiaoyang and Taher, Marwan and Qi, Xiaojuan and Davison, Andrew J},
    journal={arXiv preprint arXiv:2402.03908},
    year={2024}
    }
```
"""

with gr.Blocks() as demo:
    gr.Markdown(_HEADER_)
    # mv_images = gr.State()
    scene = gr.State(None)
    eschernet_input = gr.State(None)
    with gr.Row(variant="panel"):
        # left column
        with gr.Column():
            with gr.Row():
                input_image = gr.File(file_count="multiple")
            with gr.Row():
                run_dust3r = gr.Button("Get Pose!", elem_id="dust3r")
            with gr.Row():
                processed_image = gr.Gallery(label='Input Views', columns=2, height="100%")
            with gr.Row(variant="panel"):
                # input examples under "examples" folder
                gr.Examples(
                    examples=get_examples('examples'),
                    inputs=[input_image],
                    label="Examples (click one set of images to start!)",
                    examples_per_page=20
                )





        # right column
        with gr.Column():

            with gr.Row():
                outmodel = gr.Model3D()

            with gr.Row():
                gr.Markdown('''
                <h4><b>Check if the pose and segmentation looks correct. If not, remove the incorrect images and try again.</b></h4>
                ''')

            with gr.Row():
                with gr.Group():
                    do_remove_background = gr.Checkbox(
                        label="Remove Background", value=True
                    )
                    sample_seed = gr.Number(value=42, label="Seed Value", precision=0)

                    sample_steps = gr.Slider(
                        label="Sample Steps",
                        minimum=30,
                        maximum=75,
                        value=50,
                        step=5,
                        visible=False
                    )

                    nvs_num = gr.Slider(
                        label="Number of Novel Views",
                        minimum=5,
                        maximum=100,
                        value=30,
                        step=1
                    )

                    nvs_mode = gr.Dropdown(["archimedes circle"],   # "fixed 4 views", "fixed 8 views"
                                       value="archimedes circle", label="Novel Views Pose Chosen", visible=True)

            with gr.Row():
                gr.Markdown('''
                <h4><b>Choose your desired novel view poses number and generate! The more output images the longer it takes.</b></h4>
                ''')

            with gr.Row():
                submit = gr.Button("Submit", elem_id="eschernet", variant="primary")

            with gr.Row():
                with gr.Column():
                    output_video = gr.Video(
                        label="video", format="mp4",
                        width=379,
                        autoplay=True,
                        interactive=False
                    )

            with gr.Row():
                gr.Markdown('''The novel views are generated on an archimedean spiral. You can download the video''')

    gr.Markdown(_CITE_)

    # set dust3r parameter invisible to be clean
    with gr.Column():
        with gr.Row():
            schedule = gr.Dropdown(["linear", "cosine"],
                                       value='linear', label="schedule", info="For global alignment!", visible=False)
            niter = gr.Number(value=300, precision=0, minimum=0, maximum=5000,
                                  label="num_iterations", info="For global alignment!", visible=False)
            scenegraph_type = gr.Dropdown(["complete", "swin", "oneref"],
                                              value='complete', label="Scenegraph",
                                              info="Define how to make pairs",
                                              interactive=True, visible=False)
            same_focals = gr.Checkbox(value=True, label="Focal", info="Use the same focal for all cameras", visible=False)
            winsize = gr.Slider(label="Scene Graph: Window Size", value=1,
                                    minimum=1, maximum=1, step=1, visible=False)
            refid = gr.Slider(label="Scene Graph: Id", value=0, minimum=0, maximum=0, step=1, visible=False)

        with gr.Row():
            # adjust the confidence threshold
            min_conf_thr = gr.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20, step=0.1, visible=False)
            # adjust the camera size in the output pointcloud
            cam_size = gr.Slider(label="cam_size", value=0.05, minimum=0.01, maximum=0.5, step=0.001, visible=False)
        with gr.Row():
            as_pointcloud = gr.Checkbox(value=False, label="As pointcloud", visible=False)
            # two post process implemented
            mask_sky = gr.Checkbox(value=False, label="Mask sky", visible=False)
            clean_depth = gr.Checkbox(value=True, label="Clean-up depthmaps", visible=False)
            transparent_cams = gr.Checkbox(value=False, label="Transparent cameras", visible=False)

    # events
    # scenegraph_type.change(set_scenegraph_options,
    #                        inputs=[input_image, winsize, refid, scenegraph_type],
    #                        outputs=[winsize, refid])
    # min_conf_thr.release(fn=model_from_scene_fun,
    #                      inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
    #                              clean_depth, transparent_cams, cam_size, same_focals],
    #                      outputs=outmodel)
    # cam_size.change(fn=model_from_scene_fun,
    #                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
    #                         clean_depth, transparent_cams, cam_size, same_focals],
    #                 outputs=outmodel)
    # as_pointcloud.change(fn=model_from_scene_fun,
    #                      inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
    #                              clean_depth, transparent_cams, cam_size, same_focals],
    #                      outputs=outmodel)
    # mask_sky.change(fn=model_from_scene_fun,
    #                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
    #                         clean_depth, transparent_cams, cam_size, same_focals],
    #                 outputs=outmodel)
    # clean_depth.change(fn=model_from_scene_fun,
    #                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
    #                            clean_depth, transparent_cams, cam_size, same_focals],
    #                    outputs=outmodel)
    # transparent_cams.change(model_from_scene_fun,
    #                         inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
    #                                 clean_depth, transparent_cams, cam_size, same_focals],
    #                         outputs=outmodel)
    # run_dust3r.click(fn=recon_fun,
    #               inputs=[input_image, schedule, niter, min_conf_thr, as_pointcloud,
    #                       mask_sky, clean_depth, transparent_cams, cam_size,
    #                       scenegraph_type, winsize, refid, same_focals],
    #               outputs=[scene, outmodel, processed_image, eschernet_input])

    # events
    input_image.change(set_scenegraph_options,
                       inputs=[input_image, winsize, refid, scenegraph_type],
                       outputs=[winsize, refid])
    run_dust3r.click(fn=get_reconstructed_scene,
                     inputs=[input_image, schedule, niter, min_conf_thr, as_pointcloud,
                             mask_sky, clean_depth, transparent_cams, cam_size,
                             scenegraph_type, winsize, refid, same_focals],
                     outputs=[scene, outmodel, processed_image, eschernet_input])


    # events
    input_image.change(fn=preview_input,
                       inputs=[input_image],
                       outputs=[processed_image])

    submit.click(fn=run_eschernet,
                 inputs=[eschernet_input, sample_steps, sample_seed,
                         nvs_num, nvs_mode],
                 outputs=[output_video])



# demo.queue(max_size=10)
# demo.launch(share=True, server_name="0.0.0.0", server_port=None)
demo.queue(max_size=10).launch()

# if __name__ == '__main__':
#     main()
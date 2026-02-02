import cv2
import os
import glob
from natsort import natsorted 
from config import Config
from torchvision.transforms import Resize, ToTensor
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from attack.depth_model import import_depth_model
import numpy as np
import torch

def tensor_padding_expansion(
    patch: torch.Tensor, 
    target_size: tuple, 
    noise_range: tuple = (0, 1),
    mode: str = 'random'
) -> torch.Tensor:

    if patch.dim() == 2:
        patch = patch.unsqueeze(0)
    assert patch.dim() == 3

    C, H, W = patch.shape
    target_H, target_W = target_size
    assert target_H >= H and target_W >= W

    pad_h = target_H - H
    pad_w = target_W - W

    pad_top = torch.randint(0, pad_h + 1, (1,)).item()
    pad_bottom = pad_h - pad_top
    pad_left = torch.randint(0, pad_w + 1, (1,)).item()
    pad_right = pad_w - pad_left

    if mode == 'random':
        background = torch.empty((C, target_H, target_W), dtype=patch.dtype, device=patch.device).uniform_(*noise_range)
        background[:, pad_top:pad_top + H, pad_left:pad_left + W] = patch
        expanded = background
    elif mode == 'edge':
        import torch.nn.functional as F
        pad = (pad_left, pad_right, pad_top, pad_bottom)
        expanded = F.pad(patch, pad, mode='replicate')

    return expanded

def get_patch_area(scene_size, image_number):
    H, W = scene_size
    patch_width = 160 - image_number
    patch_height = int(patch_width / 5)
    patch_x = int(H * (0.64-image_number*0.001) - patch_height / 2)
    patch_y = int(W / 2 - patch_width / 2)
    return (patch_x, patch_y, patch_height, patch_width)

def image_patch_generate_add(): 
    video = 'video_add65_night' 
    model_name = 'monodepth2' # ['monodepth2', 'depthhints', 'DepthAnything', 'SQLdepth', 'MiDaS', 'lite_mono']
    scene_size  = Config.model_scene_sizes_WH[model_name]
    model = import_depth_model(model_name).to(Config.device).eval()

    patch = Image.open('./{}/image_patch_{}.png'.format(video, model_name)).convert('RGB')
    patch = ToTensor()(patch)

    input_folder = './{}/image'.format(video)
    output_folder_patch = './{}/patch_image/'.format(video)
    output_folder_patch_depth = './{}/patch_image_depth/'.format(video)
    output_folder_depth = './{}/image_depth/'.format(video)
    output_folder_depth_combine = './{}/image_depth_combine/'.format(video)
    if not os.path.exists(output_folder_patch):
        os.makedirs(output_folder_patch)
    if not os.path.exists(output_folder_patch_depth):
        os.makedirs(output_folder_patch_depth)
    if not os.path.exists(output_folder_depth):
        os.makedirs(output_folder_depth)
    if not os.path.exists(output_folder_depth_combine):
        os.makedirs(output_folder_depth_combine)

    images = natsorted(glob.glob(os.path.join(input_folder, '*.png')), reverse=False)
    for idx, img_path in enumerate(images):
        image = Image.open(img_path).convert('RGB')
        image = ToTensor()(image)
        if idx < 15:
            idx = idx - 25
            patch_area = get_patch_area(scene_size, int(idx))
            p_t, p_l, p_h, p_w = patch_area
            # print(p_h, p_w)
            patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
            patch_expand = tensor_padding_expansion(
                patch=patch,
                target_size=(p_h, p_w),
                mode='random'
            )
        else:
            idx = idx - 15
            patch_area = get_patch_area(scene_size, idx)
            p_t, p_l, p_h, p_w = patch_area
            patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))

            transform = transforms.Compose([
                transforms.CenterCrop((p_h, p_w))
            ])
            patch_expand = transform(patch) 

        image_patch = image.clone()
        image_patch[patch_slice] = patch_expand

        image_tensor = image.unsqueeze(0)
        image_patch_tensor = image_patch.unsqueeze(0)

        image_depth = model(image_tensor.to(Config.device))
        image_patch_depth = model(image_patch_tensor.to(Config.device))

        image_depth_save = image_depth.detach().cpu().squeeze().numpy()
        image_patch_depth_save = image_patch_depth.detach().cpu().squeeze().numpy()

        image_patch_save = transforms.ToPILImage()(image_patch.squeeze())
        patch_filename = os.path.basename(img_path)
        image_patch_save.save(output_folder_patch + 'patch_' + patch_filename)

        plt.imsave(output_folder_depth + 'depth_' + patch_filename, image_depth_save*255,  cmap='magma', vmax=255, vmin=0)
        plt.imsave(output_folder_patch_depth + 'depth_patch_' + patch_filename, image_patch_depth_save*255,  cmap='magma', vmax=255, vmin=0)

        black_line = np.zeros((10, image_depth_save.shape[1]), dtype=image_depth_save.dtype)
        image_depth_combine_save = np.vstack([image_depth_save, black_line, image_patch_depth_save])
        plt.imsave(output_folder_depth_combine + 'combine_depth_' + patch_filename, image_depth_combine_save*255,  cmap='magma', vmax=255, vmin=0)


def image_patch_generate():
    video = 'video50' # video_add  video50  video100
    model_name = 'monodepth2' # ['monodepth2', 'depthhints', 'DepthAnything', 'SQLdepth', 'MiDaS', 'lite_mono']
    scene_size  = Config.model_scene_sizes_WH[model_name]
    model = import_depth_model(model_name).to(Config.device).eval()

    patch = Image.open('./{}/image_patch_{}.png'.format(video, model_name)).convert('RGB')
    patch = ToTensor()(patch)

    input_folder = './{}/image'.format(video)
    output_folder_patch = './{}/patch_image/'.format(video)
    output_folder_patch_depth = './{}/patch_image_depth/'.format(video)
    output_folder_depth = './{}/image_depth/'.format(video)
    output_folder_depth_combine = './{}/image_depth_combine/'.format(video)
    if not os.path.exists(output_folder_patch):
        os.makedirs(output_folder_patch)
    if not os.path.exists(output_folder_patch_depth):
        os.makedirs(output_folder_patch_depth)
    if not os.path.exists(output_folder_depth):
        os.makedirs(output_folder_depth)
    if not os.path.exists(output_folder_depth_combine):
        os.makedirs(output_folder_depth_combine)

    images = natsorted(glob.glob(os.path.join(input_folder, '*.png')), reverse=False)
    for idx, img_path in enumerate(images):
        image = Image.open(img_path).convert('RGB')
        image = ToTensor()(image)
        patch_area = get_patch_area(scene_size, idx)
        p_t, p_l, p_h, p_w = patch_area
        patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))

        transform = transforms.Compose([
            transforms.CenterCrop((p_h, p_w))
        ])
        patch = transform(patch) 

        image_patch = image.clone()
        image_patch[patch_slice] = patch

        image_tensor = image.unsqueeze(0)
        image_patch_tensor = image_patch.unsqueeze(0)

        image_depth = model(image_tensor.to(Config.device))
        image_patch_depth = model(image_patch_tensor.to(Config.device))

        image_depth_save = image_depth.detach().cpu().squeeze().numpy()
        image_patch_depth_save = image_patch_depth.detach().cpu().squeeze().numpy()

        image_patch_save = transforms.ToPILImage()(image_patch.squeeze())
        patch_filename = os.path.basename(img_path)
        image_patch_save.save(output_folder_patch + 'patch_' + patch_filename)

        plt.imsave(output_folder_depth + 'depth_' + patch_filename, image_depth_save*255,  cmap='magma', vmax=255, vmin=0)
        plt.imsave(output_folder_patch_depth + 'depth_patch_' + patch_filename, image_patch_depth_save*255,  cmap='magma', vmax=255, vmin=0)

        black_line = np.zeros((10, image_depth_save.shape[1]), dtype=image_depth_save.dtype)
        image_depth_combine_save = np.vstack([image_depth_save, black_line, image_patch_depth_save])
        plt.imsave(output_folder_depth_combine + 'combine_depth_' + patch_filename, image_depth_combine_save*255,  cmap='magma', vmax=255, vmin=0)

def images_to_video_opencv():
    video = 'video_add65' 
    # image_name = ['image', 'image_depth', 'patch_image', 'patch_image_depth', 'image_depth_combine']
    # output_name = ['output_image', 'output_image_depth', 'output_patch_image', 'output_patch_image_depth', 'output_image_depth_combine']

    img_folder = './{}/patch_image'.format(video)  # ['image', 'image_depth', 'patch_image', 'patch_image_depth', 'image_depth_combine']
    output_path = './{}/output_patch_image.mp4'.format(video)  # ['output_image', 'output_image_depth', 'output_patch_image', 'output_patch_image_depth', 'output_image_depth_combine']
    duration = 8  

    images = natsorted(glob.glob(os.path.join(img_folder, '*.png')), reverse=True)
    total_frames = len(images)
    fps = total_frames / duration 

    first_frame = cv2.imread(images[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        
    try:
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not video.isOpened():
            raise Exception("error!")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        return

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read {img_path}")
            continue

        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
            
        video.write(img)
        print(f"Processing progress: {idx+1}/{total_frames} ({((idx+1)/total_frames)*100:.1f}%)", end="")

    video.release()
    print(f"The video has been saved to:{os.path.abspath(output_path)}")

if __name__ == '__main__':
    # image_patch_generate()
    image_patch_generate_add()
    # images_to_video_opencv()
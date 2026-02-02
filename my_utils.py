import torch
from PIL import Image as pil
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torchvision import transforms
import os
import random
import math

def disp_to_depth(disp,min_depth,max_depth):
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp=min_disp+(max_disp-min_disp)*disp
    depth=1/scaled_disp
    return scaled_disp,depth

def depth_to_disp(depth, min_depth, max_depth):
    scalar = 5.4
    min_disp=1/max_depth
    max_disp=1/min_depth
    scaled_disp = 1 / torch.clip(torch.clip(depth, 0, max_depth) / scalar, min_depth, max_depth)
    disp = (scaled_disp - min_disp) / (max_disp-min_disp)
    return disp

def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask=None, use_abs=False, separate=False):
    scaler=5.4
    if scene_car_mask == None:
        scene_car_mask = torch.ones_like(adv_disp1)
    dep1_adv=torch.clamp(disp_to_depth(torch.abs(adv_disp1),0.1,100)[1]*scene_car_mask*scaler,max=100)
    dep2_ben=torch.clamp(disp_to_depth(torch.abs(ben_disp2),0.1,100)[1]*scene_car_mask*scaler,max=100)
    if not separate:
        mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben))/torch.sum(scene_car_mask) if use_abs \
            else torch.sum(dep1_adv-dep2_ben)/torch.sum(scene_car_mask)
    else:
        mean_depth_diff = torch.sum(torch.abs(dep1_adv-dep2_ben), dim=[1,2,3])/torch.sum(scene_car_mask, dim=[1,2,3]) if use_abs \
            else torch.sum(dep1_adv-dep2_ben, dim=[1,2,3])/torch.sum(scene_car_mask, dim=[1,2,3])
    return mean_depth_diff

def eval_depth_diff(img1: torch.tensor, img2: torch.tensor, depth_model, filename=None, disp1=None, disp2=None):
    if disp1 == None:
        disp1 = depth_model(img1).detach().cpu().squeeze().numpy()
    else:
        disp1 = disp1.detach().cpu().squeeze().numpy()
    if disp2 == None:
        disp2 = depth_model(img2).detach().cpu().squeeze().numpy()
    else:
        disp2 = disp2.detach().cpu().squeeze().numpy()
    image1 = transforms.ToPILImage()(img1.squeeze())
    image2 = transforms.ToPILImage()(img2.squeeze())
    diff_disp = np.abs(disp1 - disp2)
    vmax = np.percentile(disp1, 95)
    
    fig: Figure = plt.figure(figsize=(12, 7))
    plt.subplot(321); plt.imshow(image1); plt.title('Image 1'); plt.axis('off')
    plt.subplot(322); plt.imshow(image2); plt.title('Image 2'); plt.axis('off')
    plt.subplot(323)
    plt.imshow(disp1, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 1'); plt.axis('off')
    plt.subplot(324)
    plt.imshow(disp2, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity 2'); plt.axis('off')
    plt.subplot(325)
    plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference'); plt.axis('off')
    plt.subplot(326)
    plt.imshow(diff_disp, cmap='magma'); plt.title('Disparity difference (scaled)'); plt.axis('off')
    fig.canvas.draw()
    if filename != None:
        plt.savefig('temp_' + filename + '.png')
    pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close()
    return pil_image, disp1, disp2

def find_neighbor(sim_graph, curr_idx, hist_score): 
    best_idx = -1
    best_metric = 0
    n = len(hist_score)
    for i in range(n):
        if i == curr_idx:
            continue
        metric = sim_graph[curr_idx][i] * hist_score[i]
        if metric > best_metric:
            best_metric = metric
            best_idx = i
    return best_idx

def normalize_score(data):
    max_positive = data[data > 0].max(initial=0)
    max_negative = data[data < 0].min(initial=0)
    normalized_data = np.zeros_like(data)
    mask_positive = data > 0
    mask_negative = data < 0
    normalized_data[mask_positive] = data[mask_positive] / max_positive if max_positive != 0 else 0
    normalized_data[mask_negative] = data[mask_negative] / abs(max_negative) if max_negative != 0 else 0

    return normalized_data

def get_patch_area(scene_size):
    H, W = scene_size
    # patch_width = 120
    patch_width = 160
    patch_height = int(patch_width / 5)
    # patch_x = int(H * 0.7 - patch_height / 2)
    patch_x = int(H * 0.64 - patch_height / 2)
    patch_y = int(W / 2 - patch_width / 2)

    return (patch_x, patch_y, patch_height, patch_width)

def get_mask_area(patch_area):
    patch_x, patch_y, patch_height, patch_width = patch_area
    errors = 24
    mask_width = patch_width + errors
    mask_height = int(patch_height * 4 + errors * 0.8)
    mask_x = int(patch_x - mask_height / 2 + errors / 4)
    # mask_x = int(patch_x - mask_height / 2 - errors / 2)
    mask_y = int(patch_y - errors / 2)
    return (mask_x, mask_y, mask_height, mask_width)

def get_patch_area_random(H, W, ratio, aspect_ratio):
    if aspect_ratio is not None:
        s_h = int(math.sqrt(H * W * ratio / aspect_ratio))
        s_w = int(aspect_ratio * s_h)
    else:
        s_h = s_w = int(math.sqrt(H * W * ratio))

    p_t = random.randint(0, H - s_h)
    p_l = random.randint(0, W - s_w)
    return (p_t, p_l, s_h, s_w)

def loss_smooth(img):
    b, c, w, h = img.shape
    img_reshaped = img.view(-1, 1, w, h)
    s1 = torch.pow(img_reshaped[:, :, 1:, :-1] - img_reshaped[:, :, :-1, :-1], 2)
    s2 = torch.pow(img_reshaped[:, :, :-1, 1:] - img_reshaped[:, :, :-1, :-1], 2)
    loss_per_img = torch.sum(s1 + s2, dim=(2, 3)) / (w*h)
    
    return torch.mean(loss_per_img.view(b, c), dim=1)
    
def loss_nps(img, color_set):
    _, c, h, w = img.shape
    color_num, c = color_set.shape
    img1 = img.unsqueeze(1)
    color_set1 = color_set.unsqueeze(2).unsqueeze(3).unsqueeze(0)
    gap = torch.min(torch.sum(torch.abs(img1 - color_set1), -1), 1).values
    return gap.sum(dim=(1, 2))/h/w

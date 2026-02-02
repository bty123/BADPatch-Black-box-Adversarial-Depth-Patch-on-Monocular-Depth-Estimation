from config import Config
from torchvision.transforms import Resize, ToTensor
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from attack.depth_model import import_depth_model

def get_patch_area(scene_size):
    H, W = scene_size
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

attack_name = 'hardbeat' # ['blackbox', 'whitebox', 'badPart', 'sparse_RS', 'hardbeat']
model_name = 'monodepth2' # ['monodepth2', 'depthhints', 'DepthAnything', 'SQLdepth', 'MiDaS', 'lite_mono']
scene_size  = Config.model_scene_sizes_WH[model_name]
scene_H, scene_W = scene_size
model = import_depth_model(model_name).to(Config.device).eval()

weather = 'night' # ['default', 'cloudy', 'night', 'rainy']
image = Image.open('./weather_image/image_{}.png'.format(weather)).convert('RGB')
image = ToTensor()(image)
patch = Image.open('./weather_image/image_patch_{}.png'.format(attack_name)).convert('RGB')
patch = ToTensor()(patch)
standard_image = Image.open('./weather_image/image_default.png').convert('RGB')
standard_image = ToTensor()(standard_image)

patch_area = get_patch_area(scene_size)
p_t, p_l, p_h, p_w = patch_area
patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))

image_patch = image.clone()
image_patch[patch_slice] = patch

image_tensor = image.unsqueeze(0)
image_patch_tensor = image_patch.unsqueeze(0)
standard_image_tensor = standard_image.unsqueeze(0)

image_depth = model(image_tensor.to(Config.device))
image_patch_depth = model(image_patch_tensor.to(Config.device))
standard_image_depth = model(standard_image_tensor.to(Config.device))

col_index = 400
depth_column = image_depth[:, :, :, col_index:col_index+1] 
standard_depth = depth_column.expand(-1, -1, -1, image_depth.size(3))

image_depth_save = image_depth.detach().cpu().squeeze().numpy()
image_patch_depth_save = image_patch_depth.detach().cpu().squeeze().numpy()
standard_depth_save = standard_depth.detach().cpu().squeeze().numpy()

image_patch_save = transforms.ToPILImage()(image_patch.squeeze())
image_patch_save.save('./weather_image/{}/image_{}_patch.png'.format(attack_name, weather))

plt.imsave('./weather_image/{}/image_{}_depth.png'.format(attack_name, weather), image_depth_save*255,  cmap='magma', vmax=255, vmin=0)
plt.imsave('./weather_image/{}/image_{}_patch_depth.png'.format(attack_name, weather), image_patch_depth_save*255,  cmap='magma', vmax=255, vmin=0)
plt.imsave('./weather_image/standard_depth.png', standard_depth_save*255,  cmap='magma', vmax=255, vmin=0)

m_t, m_l, m_h, m_w = get_mask_area(patch_area)
scene_mask = torch.zeros((1, 1, scene_H, scene_W)).to(Config.device)
scene_mask[:, :, m_t : m_t + m_h, m_l : m_l + m_w] = 1  
        
env_loss = torch.sum(torch.abs(image_patch_depth-image_depth)*(1-scene_mask))/torch.sum(image_depth*(1-scene_mask))
dis_loss = torch.sum(torch.abs(image_patch_depth-standard_depth)*scene_mask)/torch.sum(standard_depth*scene_mask)

file_name = './weather_image/{}/result_{}.txt'.format(attack_name, weather)
with open(file_name, 'w') as f:
    f.write(f'env_loss={env_loss.item()}, dis_loss={dis_loss.item()}\n')








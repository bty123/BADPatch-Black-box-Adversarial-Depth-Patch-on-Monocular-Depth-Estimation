from config import Config
from torchvision.transforms import Resize, ToTensor
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from attack.depth_model import import_depth_model

def get_patch_area(scene_size, distance):
    if distance == '2.5':
        H, W = scene_size
        patch_width = 160
        patch_height = int(patch_width / 5)
        # patch_x = int(H * 0.7 - patch_height / 2)
        patch_x = int(H * 0.64 - patch_height / 2)
        patch_y = int(W / 2 - patch_width / 2)
        return (patch_x, patch_y, patch_height, patch_width)
    elif distance == '2.75':
        H, W = scene_size
        patch_width = 140
        patch_height = int(patch_width / 5)
        # patch_x = int(H * 0.7 - patch_height / 2)
        patch_x = int(H * 0.62 - patch_height / 2)
        patch_y = int(W / 2 - patch_width / 2)
        return (patch_x, patch_y, patch_height, patch_width)
    elif distance == '3':
        H, W = scene_size
        patch_width = 125
        patch_height = int(patch_width / 5)
        # patch_x = int(H * 0.7 - patch_height / 2)
        patch_x = int(H * 0.61 - patch_height / 2)
        patch_y = int(W / 2 - patch_width / 2)
        return (patch_x, patch_y, patch_height, patch_width)
    else:
        raise RuntimeError(f"distance size undefined!")

model_name = 'monodepth2' # ['monodepth2', 'depthhints', 'DepthAnything', 'SQLdepth', 'MiDaS', 'lite_mono']
scene_size  = Config.model_scene_sizes_WH[model_name]
model = import_depth_model(model_name).to(Config.device).eval()

distance = '3'
image = Image.open('./distance_image/image_{}.png'.format(distance)).convert('RGB')
image = ToTensor()(image)
patch = Image.open('./distance_image/image_patch_{}.png'.format(model_name)).convert('RGB')
patch = ToTensor()(patch)

patch_area = get_patch_area(scene_size, distance)
p_t, p_l, p_h, p_w = patch_area
patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))

patch_resize = Resize((p_h, p_w))
patch = patch_resize(patch)

image_patch = image.clone()
image_patch[patch_slice] = patch

image_tensor = image.unsqueeze(0)
image_patch_tensor = image_patch.unsqueeze(0)

image_depth = model(image_tensor.to(Config.device))
image_patch_depth = model(image_patch_tensor.to(Config.device))

image_depth_save = image_depth.detach().cpu().squeeze().numpy()
image_patch_depth_save = image_patch_depth.detach().cpu().squeeze().numpy()

image_patch_save = transforms.ToPILImage()(image_patch.squeeze())
image_patch_save.save('./distance_image/image_{}_patch.png'.format(distance))

plt.imsave('./distance_image/image_{}_depth.png'.format(distance), image_depth_save*255,  cmap='magma', vmax=255, vmin=0)
plt.imsave('./distance_image/image_{}_patch_depth.png'.format(distance), image_patch_depth_save*255,  cmap='magma', vmax=255, vmin=0)




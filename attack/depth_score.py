from config import Config
import torch
import numpy as np
import os
from torchvision import transforms
from PIL import Image as pil
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from my_utils import loss_smooth, loss_nps, get_mask_area

class DepthScore(object):
    def __init__(self, model, model_name, img_set, n_batch, batch_size, patch_area) -> None:
        self.model = model
        self.model_name = model_name
        self.img_set = img_set
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.query_times = 0
        self.patch_area = patch_area
        p_t, p_l, p_h, p_w = patch_area
        self.patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
        self.patch_slice1 = (slice(0, self.batch_size), slice(0, 3), slice(p_t, p_t + p_h), slice(p_l, p_l + p_w))
        self.ori_disp = []
        self.disp_target = torch.zeros(1).float().to(Config.device)
        train_batch, c, self.scene_H, self.scene_W = self.img_set[0].shape  
        self.scene_size = (self.scene_H, self.scene_W)

        with torch.no_grad():
            for i in range(self.n_batch):
                scene = self.img_set[i]
                scene_copy = scene.clone()     
                disp_ref = self.model(scene_copy.to(Config.device))
                self.ori_disp.append(disp_ref)

    def numpy2tensor(self, x):
        if len(x.shape) == 3:
            x = torch.from_numpy(x).unsqueeze(0)
        elif len(x.shape) == 4:
            x = torch.from_numpy(x)
        return x
    
    def disp_diff_compute(self, patch):
        with torch.no_grad():
            patch = self.numpy2tensor(patch)
            score_map = []
            new_slice = (slice(0, self.batch_size),) + self.patch_slice
            for i in range(self.n_batch):
                scene = self.img_set[i].clone()
                scene[new_slice] = patch
                disp = self.model(scene.to(Config.device)) 
                prob_map = self.ori_disp[i] - disp
                score_map.append(prob_map)
            score_map = torch.cat(score_map, dim=0)
            score_map = score_map.mean(dim=0, keepdim=True)

        return score_map.squeeze().detach().cpu().numpy()
    
    def score_creteria(self, disp_ori, est_disp, mask, new_standard_disp, separate=True):
        Ablation = 'ldle' # ld le ldle
        if separate:
            # score_mask = torch.sum(torch.abs(est_disp - self.disp_target) * mask, dim=[1, 2, 3]) / torch.sum(mask, dim=[1,2,3])
            score_mask_ori = -torch.sum(0.1 * torch.pow((est_disp - disp_ori) * mask, 2), dim=[1, 2, 3]) / torch.sum(mask, dim=[1,2,3])
            if Ablation == 'ld':
                score_mask = torch.sum(1*torch.pow((est_disp - new_standard_disp) * mask, 2), dim=[1, 2, 3]) / torch.sum(mask, dim=[1,2,3])
                return score_mask_ori + score_mask
            elif Ablation == 'le':
                score_inverse_mask = torch.sum(1*torch.pow((est_disp - disp_ori) * (1 - mask), 2), dim=[1, 2, 3]) / torch.sum(1 - mask, dim=[1,2,3])
                return score_mask_ori + score_inverse_mask
            elif Ablation == 'ldle':
                score_mask = torch.sum(1*torch.pow((est_disp - new_standard_disp) * mask, 2), dim=[1, 2, 3]) / torch.sum(mask, dim=[1,2,3])
                score_inverse_mask = torch.sum(1*torch.pow((est_disp - disp_ori) * (1 - mask), 2), dim=[1, 2, 3]) / torch.sum(1 - mask, dim=[1,2,3])
                return score_mask_ori + score_mask + score_inverse_mask
            
        else:
            # score_mask = torch.sum(torch.abs(est_disp - self.disp_target) * mask) / torch.sum(mask)
            score_mask_ori = -torch.sum(0.1 * torch.pow((est_disp - disp_ori) * mask, 2)) / torch.sum(mask)
            if Ablation == 'ld':
                score_mask = torch.sum(1*torch.pow((est_disp - new_standard_disp) * mask, 2)) / torch.sum(mask)
                return score_mask_ori + score_mask
            elif Ablation == 'le':
                score_inverse_mask = torch.sum(1*torch.pow((est_disp - disp_ori) * (1 - mask), 2)) / torch.sum(1 - mask)
                return score_mask_ori + score_inverse_mask
            elif Ablation == 'ldle':
                score_mask = torch.sum(1*torch.pow((est_disp - new_standard_disp) * mask, 2)) / torch.sum(mask)
                score_inverse_mask = torch.sum(1*torch.pow((est_disp - disp_ori) * (1 - mask), 2)) / torch.sum(1 - mask)
                return score_mask_ori + score_mask + score_inverse_mask

    def get_absolute_error(self, disp, new_standard_disp, ori_disp):
        m_t, m_l, m_h, m_w = get_mask_area(self.patch_area)
        scene_mask = torch.zeros((1, 1, self.scene_H, self.scene_W)).to(Config.device)
        scene_mask[:, :, m_t : m_t + m_h, m_l : m_l + m_w] = 1  
        
        env_loss = torch.sum(torch.abs(disp-ori_disp)*(1-scene_mask))/torch.sum(ori_disp*(1-scene_mask))
        dis_loss = torch.sum(torch.abs(disp-new_standard_disp)*scene_mask)/torch.sum(new_standard_disp*scene_mask)

        return dis_loss, env_loss

    def score(self, patch, mask, new_standard_disp, epe=False):
        with torch.no_grad():
            if isinstance(patch, np.ndarray):  
                patch = self.numpy2tensor(patch)  
            loss_list = []
            if epe:  
                env_loss_list = []
                dis_loss_list = []

            new_slice = (slice(0, self.batch_size),) + self.patch_slice  
            for i in range(self.n_batch):
                scene = self.img_set[i].clone()  
                scene[new_slice] = patch  
                disp = self.model(scene.to(Config.device)) 

                if epe: 
                    dis_loss, env_loss = self.get_absolute_error(disp, new_standard_disp, self.ori_disp[i])
                    dis_loss_list.append(dis_loss.item())
                    env_loss_list.append(env_loss.item())
                else:
                    loss = self.score_creteria(self.ori_disp[i], disp, mask, new_standard_disp, separate=False).item()
                    loss_list.append(loss)
        if epe:
            return np.mean(dis_loss_list), np.mean(env_loss_list)
        else:
            return np.mean(loss_list)

    def get_gredient(self, patches_candi, patch, scene, mask, new_standard_disp):
        with torch.no_grad():
            _, _, w, h = patches_candi.shape
            color_set = torch.tensor([[0,0,0],[255,255,255],[0,18,79],[5,80,214],
                                  [71,178,243],[178,159,211],[77,58,0],[211,191,167],
                                  [247,110,26],[110,76,16]]).to(Config.device).float() / 255
            patches_candi = self.numpy2tensor(patches_candi).to(Config.device)
            patch = self.numpy2tensor(patch).to(Config.device)
            trail = patches_candi.shape[0]
            scenes = scene.repeat(trail, 1, 1, 1)
            disp_ori = self.model(scene.to(Config.device))
            
            new_slice1 = (slice(0,1),) + self.patch_slice
            scene[new_slice1] = patch
            disp_ref = self.model(scene.to(Config.device))

            new_slice2 = (slice(0,trail),) + self.patch_slice
            scenes[new_slice2] = patches_candi

            disp = self.model(scenes.to(Config.device))
            
            nps_loss_candi = loss_nps(patches_candi, color_set) / (w * h) 
            nps_loss_curr = loss_nps(patch, color_set) / (w * h) 

            tv_loss_candi = loss_smooth(patches_candi) / (w * h) 
            tv_loss_curr = loss_smooth(patch) / (w * h) 

            score_ref = self.score_creteria(disp_ori, disp_ref, mask, new_standard_disp) 
            score_candi = self.score_creteria(disp_ori, disp, mask, new_standard_disp) 

            Ablation1 = 'ldlenpstv' # ldle ldlenpstv 
            if False:  # False True
                scores_disp = (score_ref - score_candi) / (score_ref + 1e-8)
                scores_nps = (nps_loss_curr - nps_loss_candi) / (nps_loss_curr + 1e-8)
                scores_tv = (tv_loss_curr - tv_loss_candi) / (tv_loss_curr + 1e-8)

                alpha_nps = torch.std(scores_disp) / (torch.std(scores_nps) + 1e-8)
                alpha_tv = torch.std(scores_disp) / (torch.std(scores_tv) + 1e-8)
                
                gredient = scores_disp + alpha_nps * scores_nps + alpha_tv * scores_tv
            else:
                scores_disp = score_ref - score_candi
                scores_nps = nps_loss_curr - nps_loss_candi
                scores_tv = tv_loss_curr - tv_loss_candi
                if Ablation1 == 'ldlenpstv':
                    gredient = scores_disp + scores_nps*10 + scores_tv*10
                elif Ablation1 == 'ldle':
                    gredient = scores_disp
            
        return gredient


    def viz(self, patch):
        with torch.no_grad():
            scene = self.img_set[0][0:1,:,:,:].clone() 
            disp_ref = self.ori_disp[0][0:1,:,:,:].clone() 
            disp_ori= self.model(scene.to(Config.device))
            new_slice = (slice(0, 1),) + self.patch_slice
            scene_patched = scene.clone()
            scene_patched[new_slice] = patch
            disp = self.model(scene_patched.to(Config.device))

        disp_ori = disp_ori.detach().cpu().squeeze().numpy()
        disp_ref = disp_ref.detach().cpu().squeeze().numpy()
        disp = disp.detach().cpu().squeeze().numpy()

        scene = transforms.ToPILImage()(scene.squeeze())
        scene_patched = transforms.ToPILImage()(scene_patched.squeeze())
        diff_disp = np.abs(disp_ori - disp)
        diff_disp_patched = np.abs(disp_ref - disp)
        vmax = np.percentile(disp_ori, 95)

        fig: Figure = plt.figure(figsize=(12, 7)) 
        plt.subplot(321); plt.imshow(scene); plt.title('original scene'); plt.axis('off')
        plt.subplot(322); plt.imshow(scene_patched); plt.title('patched scene'); plt.axis('off')
        plt.subplot(323)
        plt.imshow(disp_ori, cmap='magma', vmax=vmax, vmin=0); plt.title('original disparity'); plt.axis('off')
        plt.subplot(324)
        plt.imshow(disp, cmap='magma', vmax=vmax, vmin=0); plt.title('attacked disparity'); plt.axis('off')
        plt.subplot(325)
        plt.imshow(diff_disp, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference (ori)'); plt.axis('off')
        plt.subplot(326)
        plt.imshow(diff_disp_patched, cmap='magma', vmax=vmax, vmin=0); plt.title('Disparity difference (patched)'); plt.axis('off')
        fig.canvas.draw()
        pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)

        return pil_image
    
    def viz_test(self, patch, patch_path, stop):
        for i in range(stop):
        
            with torch.no_grad():
                scene = self.img_set[i].clone() 
                disp_ref = self.ori_disp[i].clone() 
                disp_ori= self.model(scene.to(Config.device))
                scene_patched = scene.clone()
                scene_patched[self.patch_slice1] = patch
                disp = self.model(scene_patched.to(Config.device))
            disp_ori = disp_ori.detach().cpu().squeeze().numpy()
            disp_ref = disp_ref.detach().cpu().squeeze().numpy()
            disp = disp.detach().cpu().squeeze().numpy()

            for j in range(self.batch_size):
                scene1 = transforms.ToPILImage()(scene[j].squeeze())
                scene_patched1 = transforms.ToPILImage()(scene_patched[j].squeeze())
                diff_disp1 = np.abs(disp_ori[j] - disp[j])
                diff_disp_patched1 = np.abs(disp_ref[j] - disp[j])
                vmax = np.percentile(disp_ori[j], 100)
                fig: Figure = plt.figure(figsize=(12, 7)) 
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.2)
                plt.subplot(321); plt.imshow(scene1); plt.title('original scene'); plt.axis('off')
                plt.subplot(322); plt.imshow(scene_patched1); plt.title('patched scene'); plt.axis('off')
                plt.subplot(323)
                plt.imshow(disp_ori[j]*255, cmap='magma', vmax=255, vmin=0); plt.title('original disparity'); plt.axis('off')
                plt.subplot(324)
                plt.imshow(disp[j]*255, cmap='magma', vmax=255, vmin=0); plt.title('attacked disparity'); plt.axis('off')
                plt.subplot(325)
                plt.imshow(diff_disp1*255, cmap='magma', vmax=255, vmin=0); plt.title('Disparity difference (ori)'); plt.axis('off')
                plt.subplot(326)
                plt.imshow(diff_disp_patched1*255, cmap='magma', vmax=255, vmin=0); plt.title('Disparity difference (patched)'); plt.axis('off')
                fig.canvas.draw()
                pil_image = pil.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                plt.close(fig)

                pil_image_dir =  os.path.join(patch_path, 'image_test')
                os.makedirs(pil_image_dir, exist_ok=True)

                pil_image.save('./{}/image_test/image_test_{}_{}.png'.format(patch_path, i, j))
        
        with torch.no_grad():
            scene_save = self.img_set[1].clone()
            disp_ori_save= self.model(scene.to(Config.device))
            scene_patched_save = scene_save.clone()
            scene_patched_save[self.patch_slice1] = patch
            disp_patch_save = self.model(scene_patched.to(Config.device))
        disp_ori_save = disp_ori_save.detach().cpu().squeeze().numpy()
        disp_patch_save = disp_patch_save.detach().cpu().squeeze().numpy()
        scene1_save = transforms.ToPILImage()(scene[3].squeeze())
        scene_patched1_save = transforms.ToPILImage()(scene_patched[3].squeeze())
        diff_disp_save = np.abs(disp_ori_save[3] - disp_patch_save[3])

        scene1_save.save('./{}/scene_ori.png'.format(patch_path))
        scene_patched1_save.save('./{}/scene_patch.png'.format(patch_path))
        
        plt.imsave('./{}/depth_ori.png'.format(patch_path), disp_ori_save[3]*255,  cmap='magma', vmax=255, vmin=0)
        plt.imsave('./{}/depth_patch.png'.format(patch_path), disp_patch_save[3]*255, cmap='magma', vmax=255, vmin=0)
        plt.imsave('./{}/depth_diff.png'.format(patch_path), diff_disp_save*255, cmap='magma', vmax=255, vmin=0)


    


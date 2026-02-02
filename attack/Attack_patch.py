import torch
import numpy as np
from numpy.linalg import norm
import math
from torch.utils.data.dataloader import DataLoader
from scipy.special import softmax
from config import Config
from my_utils import normalize_score, find_neighbor, loss_smooth, loss_nps, get_mask_area
from PIL import Image
from attack.dataset import KittiDataset, CarlaDataset, CarlaOwnDataset

class Attack_patch():
    def __init__(self, model, model_name, patch_area, n_batch=1, batch_size=5, trail=20):
        self.model = model
        self.model_name = model_name
        self.patch_area = patch_area
        p_t, p_l, p_h, p_w = patch_area
        self.patch_slice = (slice(0, 3), slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))
        self.n_batch = n_batch
        self.batch_size = batch_size
        self.trail = trail
        self.train_eval_set = []
        self.test_set = []

        self.train_dataset = CarlaOwnDataset(self.model_name, main_dir=Config.Carla_own_dataset_root, mode='training')
        self.test_dataset = CarlaOwnDataset(self.model_name, main_dir=Config.Carla_own_dataset_root, mode='testing')
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
        from attack.depth_score import DepthScore
        for i, (scenes, _) in enumerate(self.train_loader):
            self.train_eval_set.append(scenes)
        for i, (scenes, _) in enumerate(self.test_loader):
            self.test_set.append(scenes)
        self.Score = DepthScore(model, model_name, self.train_eval_set, n_batch, batch_size, patch_area)
        self.eval_Score = DepthScore(model, model_name, self.test_set, n_batch, batch_size, patch_area)

    def p_selection(self, p_init, it, n_iters, version='v6'):
        it = int(it / n_iters * 10000)
        if version == 'v1':
            sche = [10, 50, 200, 500, 1000, 2000, 4000, 6000, 8000, 10000]
        elif version == 'v2':
            sche = [100, 500, 1200, 2000, 4000, 8000, 9000, 9500, 9800, 10000]
        elif version == 'v3':
            return p_init
        elif version == 'v4':
            sche = [4000, 7000, 9000, 10000]
        elif version == 'v5':
            sche = [100, 250, 500, 1000, 3000]
        elif version == 'v6':
            sche = [100, 500, 1500, 3000, 5000, 10000]
        elif version == 'v7':
            sche = [100, 500, 3000, 5000, 10000]
        elif version == 'v8':
            sche = [100, 500, 4000, 8000, 10000]
        else:
            raise NotImplementedError("Schedule not implemented.")
        
        if it <= sche[0]:
            p = p_init
        elif sche[0] < it <= sche[1]:
            p = p_init / 2
        elif sche[1] < it <= sche[2]:
            p = p_init / 4
        elif sche[2] < it <= sche[3]:
            p = p_init / 8
        elif sche[3] < it <= sche[4]:
            p = p_init / 16
        elif sche[4] < it <= sche[5]:
            p = p_init / 32
        elif sche[5] < it <= sche[6]:
            p = p_init / 64
        elif sche[6] < it <= sche[7]:
            p = p_init / 128
        elif sche[7] < it <= sche[8]:
            p = p_init / 256
        elif sche[8] < it <= sche[9]:
            p = p_init / 512
        else:
            raise NotImplementedError("No such interval.")

        return p

    def create_patch(self, params, p_h, p_w, mode):
        if mode == 'square':
            params_np = np.array(params).reshape((-1, 6))
            params_np = params_np[params_np[:, 0].argsort()[::-1]]
            patch = np.zeros([3, p_h, p_w])
            max_edge_r = 0.3
            for i in range(params_np.shape[0]):
                s = max(1, math.ceil(max_edge_r * min(p_h, p_w) * params_np[i, 0]))
                h_start = int((p_h-s) * params_np[i, 1])
                w_start = int((p_w-s) * params_np[i, 2])
                patch[:, h_start:h_start + s, w_start:w_start + s] = np.reshape(params_np[i, 3:6], (3, 1, 1))
        elif mode == 'pixel':
            patch = np.array(params).reshape((3, p_h, p_w))
        return patch

    def numpy2tensor(self, x):
        if len(x.shape) == 3:
            x = torch.from_numpy(x).unsqueeze(0)
        elif len(x.shape) == 4:
            x = torch.from_numpy(x)
        return x

    def blackbox_attack(self, trail, test_name):
        beta1 = Config.beta1 
        beta2 = Config.beta2 
        lr = Config.lr 
        eps = Config.eps
        Adaptive_adjustment = False 
        Gradient_Normalization = False 

        scene_loader_iter = iter(self.train_loader)  
        scenes = next(scene_loader_iter)  
        train_batch, c, scene_H, scene_W = scenes.shape  
        p_t, p_l, p_h, p_w = self.patch_area  
        patch = np.random.rand(c, p_h, p_w)
        noise_weight = 0.1

        col_index = 400
        scene_standard = scenes[0:1]  
        standard_disp = self.model(scene_standard.to(Config.device))
        depth_column = standard_disp[:, :, :, col_index:col_index+1]
        new_standard_disp = depth_column.expand(-1, -1, -1, standard_disp.size(3))

        m_t, m_l, m_h, m_w = get_mask_area(self.patch_area)
        mask = torch.zeros((1, 1, scene_H, scene_W)).to(Config.device)
        mask[:, :, m_t : m_t + m_h, m_l : m_l + m_w] = 1  
        
        best_loss = 100
        early_stop_step = 0
        result_list = []
        # Train
        print('===============================')
        print("Start training ...")
        for epoch in range(1000):
            deltas = []
            avg_mean = []
            for i, (scenes, _) in enumerate(self.train_loader):
                noise = np.random.choice([0, 1], size=[trail, *patch.shape])  
                noise = 2.0 * (noise - 0.5) * noise_weight 

                patches = patch.copy()
                patches = patches[None, ...]  
                patches = np.repeat(patches, trail, axis=0)

                patches_candi = np.clip(noise + patches, 0, 1)  
                noise = patches_candi - patches  
                indice = torch.randperm(Config.scenes_batch) 
                scene = scenes[indice] 

                scores_list = []  
                for num in range(Config.scenes_batch):
                    scene = scenes[indice[num:num+1]]
                    score = self.Score.get_gredient(patches_candi, patch, scene, mask, new_standard_disp)
                    scores_list.append(score)  

                scores_stacked = torch.stack(scores_list)
                scores_mean = torch.mean(scores_stacked, dim=0)

                if Gradient_Normalization:
                    candi_y = normalize_score(scores_mean.cpu().numpy())
                    if np.sum(np.abs(candi_y)) == 0:
                        candi_y = np.ones_like(candi_y)
                else:
                    candi_y = scores_mean.cpu().numpy()
                mean_y = np.mean(candi_y) 
                avg_mean.append(mean_y)

                if Adaptive_adjustment: 
                    pos_cnt = np.sum(candi_y > 0)
                    neg_cnt = trail - pos_cnt
                    candi_y[candi_y > 0] *= (2/(pos_cnt + eps))
                    candi_y[candi_y <= 0] *= (2/(neg_cnt + eps))
                    delta = np.mean(noise * candi_y.reshape((trail, 1, 1, 1)), axis=0)
                else:
                    delta = np.mean(noise * candi_y.reshape((trail, 1, 1, 1)), axis=0)

                if np.isnan(delta).any() or np.isinf(delta).any():
                    print("NaN or inf")

                if norm(delta) == 0:
                    print("norm == 0")
                    
                delta = delta / norm(delta)
                deltas.append(delta)
            delta = np.mean(deltas, axis=0)
            gradf = torch.from_numpy(delta)

            gradf_flat = gradf.flatten()
            if epoch == 0:
                grad_momentum = gradf
                full_matrix   = torch.outer(gradf_flat, gradf_flat)
            else:
                grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
                full_matrix   = beta2 * full_matrix + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)
            grad_momentum /= (1 - beta1 ** (epoch + 1))
            full_matrix   /= (1 - beta2 ** (epoch + 1))
            factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
            gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)

            patch = patch + lr * gradf.numpy()   
            patch = np.clip(patch, 0, 1)
            loss_curr = self.Score.score(patch, mask, new_standard_disp)
            if best_loss > loss_curr.item():
                best_loss = loss_curr.item()
                best_patch_curr = patch
                early_stop_step = 0
            else:
                early_stop_step += 1
            print(f'Epoch {epoch}, Loss: {loss_curr.item()}')

            dis, env = self.Score.score(patch, mask, new_standard_disp, epe=True)
            result_list.append((loss_curr, dis, env))
            print(f'train dis: {dis}, train env: {env}')
            if early_stop_step >= 5:
                print('early stop!')
                break
 
        loss_curr = self.eval_Score.score(best_patch_curr, mask, new_standard_disp)
        print(f'Loss: {loss_curr.item()}')
        eval_dis, eval_env = self.eval_Score.score(best_patch_curr, mask, new_standard_disp, epe=True)
        print(f'evaluation dis: {eval_dis}, evaluation env: {eval_env}')

        file_name = './logs/{}/result.txt'.format(test_name)
        with open(file_name, 'w') as f:
            for i, (loss, dis, env) in enumerate(result_list):
                f.write(f'epoch {i}: loss={loss}, dis={dis}, env={env}\n')
            f.write(f'-----------------------------------------------\n')
            f.write(f'test: loss={loss_curr}, eval_dis={eval_dis}, eval_env={eval_env}\n')

        best_patch = torch.Tensor(best_patch_curr)
        patch_path = './logs/{}'.format(test_name)
        self.eval_Score.viz_test(best_patch, patch_path, 2)
        scene_image = self.eval_Score.viz(best_patch)
        scene_image.save('./logs/{}/image_scene.png'.format(test_name))

        patch_best = np.transpose(best_patch_curr, (1, 2, 0))
        patch_image = Image.fromarray((patch_best * 255).astype(np.uint8))
        patch_image.save('./logs/{}/image_patch.png'.format(test_name))
        torch.save(best_patch, './logs/{}/best_patch.pt'.format(test_name))

    def whitebox_attack(self, test_name):
        scene_loader_iter = iter(self.train_loader)  
        scenes = next(scene_loader_iter)  
        color_set = torch.tensor([[0,0,0],[255,255,255],[0,18,79],[5,80,214],[71,178,243],[178,159,211],[77,58,0],[211,191,167],[247,110,26],[110,76,16]]).to(Config.device).float() / 255
        train_batch, c, scene_H, scene_W = scenes.shape  
        p_t, p_l, p_h, p_w = self.patch_area  
        
        patch_curr = np.random.randn(c, p_h, p_w)
        patch_curr = self.numpy2tensor(patch_curr)
        patch_curr = patch_curr.to(Config.device)
            
        col_index = 400
        scene_standard = scenes[0:1]
        standard_disp = self.model(scene_standard.to(Config.device))
        depth_column = standard_disp[:, :, :, col_index:col_index+1]
        new_standard_disp = depth_column.expand(-1, -1, -1, standard_disp.size(3))
        
        patch_curr.requires_grad_(True)
        optimizer = torch.optim.Adam([patch_curr], lr=Config.lr, betas=(Config.beta1, Config.beta2), eps=Config.eps)
        best_loss = 100
        early_stop_step = 0
        result_list = []
        print('===============================')
        print("Start training ...")
        for epoch in range(100):
            for i, (scenes, _) in enumerate(self.train_loader):
                optimizer.zero_grad()
                patch_curr.data = torch.clip(patch_curr.data, 0, 1)
                indice = torch.randperm(Config.scenes_batch)
                scene = scenes[indice]
                ori_disp = self.model(scene.to(Config.device))

                b, _, h, w = scene.shape
                new_slice = (slice(0, b),) + self.patch_slice
                scene_patched = scene.clone().detach()
                scene_patched[new_slice] = patch_curr
                est_disp = self.model(scene_patched.to(Config.device))

                m_t, m_l, m_h, m_w = get_mask_area(self.patch_area)
                mask = torch.zeros_like(est_disp)
                mask[:, :, m_t:m_t+m_h, m_l:m_l+m_w] = 1  
                inverse_mask = 1 - mask

                disp_loss_non_mask = torch.sum(10 * torch.pow((est_disp - ori_disp) * inverse_mask, 2)) / torch.sum(inverse_mask)
                disp_loss = torch.sum(10 * torch.pow((est_disp - new_standard_disp) * mask, 2)) / torch.sum(mask)

                nps_loss = loss_nps(patch_curr, color_set) / p_h * p_w
                tv_loss = loss_smooth(patch_curr) / p_h * p_w
                loss = disp_loss + disp_loss_non_mask + nps_loss + tv_loss

                loss.backward()
                optimizer.step()

            if best_loss > loss.item():
                best_loss = loss.item()
                best_patch_curr = patch_curr
                early_stop_step = 0
            else:
                early_stop_step += 1

            print(f'Epoch {epoch}, Loss: {loss.item()}')
            print('disp_loss_non_mask:', disp_loss_non_mask.item())
            print('disp_loss:', disp_loss.item())
            print('nps_loss:', nps_loss.item())
            print('tv_loss:', tv_loss.item())

            dis, env = self.Score.score(patch_curr, mask, new_standard_disp, epe=True)
            result_list.append((loss.item(), dis, env))
            print(f'train dis: {dis}, train env: {env}')
            if early_stop_step >= 5:
                print('early stop!')
                break
        
        eval_dis, eval_env = self.eval_Score.score(best_patch_curr, mask, new_standard_disp, epe=True)
        print(f'evaluation dis: {eval_dis}, evaluation env: {eval_env}')

        file_name = './logs/{}/result.txt'.format(test_name)
        with open(file_name, 'w') as f:
            for i, (loss, dis, env) in enumerate(result_list):
                f.write(f'epoch {i}: loss={loss}, dis={dis}, env={env}\n')
            f.write(f'-----------------------------------------------\n')
            f.write(f'test: eval_dis={eval_dis}, eval_env={eval_env}\n')

        best_patch = torch.Tensor(best_patch_curr)
        patch_path = './logs/{}'.format(test_name)
        self.eval_Score.viz_test(best_patch, patch_path, 2)
        scene_image = self.eval_Score.viz(best_patch)
        scene_image.save('./logs/{}/image_scene.png'.format(test_name))
        patch_curr_np = best_patch_curr.cpu().detach().squeeze().numpy()  
        patch_best = np.transpose(patch_curr_np, (1, 2, 0))
        patch_image = Image.fromarray((patch_best * 255).astype(np.uint8))
        patch_image.save('./logs/{}/image_patch.png'.format(test_name))
        torch.save(best_patch, './logs/{}/best_patch.pt'.format(test_name))

    def BadPart(self, alpha=0.1, n_iters=10000, init_iters=100, p_init=0.025, p_sche='v6', square_steps=200, num_pos=1, trail=10, test_name = '1'):
        scene_loader_iter = iter(self.train_loader)  
        scenes = next(scene_loader_iter)  
        train_batch, c, scene_H, scene_W = scenes.shape  

        p_t, p_l, p_h, p_w = self.patch_area  
        init_trail = trail  
        n_patch_features = c * p_h * p_w  
        patch_curr = np.random.choice([-alpha, alpha], size=[c, 1, p_w]) 
        patch_curr = np.repeat(patch_curr, p_h, axis=1) 
        patch_curr += 0.5   
        patch_curr = np.clip(patch_curr, 0, 1) 

        col_index = 400
        scene_standard = scenes[0:1]     
        standard_disp = self.model(scene_standard.to(Config.device))
        depth_column = standard_disp[:, :, :, col_index:col_index+1]
        new_standard_disp = depth_column.expand(-1, -1, -1, standard_disp.size(3))

        m_t, m_l, m_h, m_w = get_mask_area(self.patch_area)
        mask = torch.zeros((1, 1, scene_H, scene_W)).to(Config.device)
        mask[:, :, m_t : m_t + m_h, m_l : m_l + m_w] = 1  

        patch_best = patch_curr.copy()  
        best_loss = self.Score.score(patch_best, mask, new_standard_disp)
        print(f'initial_score: {best_loss}')

        stable_count = 0 
        stable_count_inSquare = 0 
        eps = Config.eps
        init_noise_weight = Config.init_noise_weight
        min_noise_weight = Config.min_noise_weight 
        noise_weight = init_noise_weight
        threshold_betwSquare = Config.threshold_betwSquare[self.model_name] 
        threshold_inSquare = Config.threshold_inSquare[self.model_name] 
        lr = Config.lr
        beta1 = Config.beta1
        beta2 = Config.beta2
        minus_mean = Config.minus_mean 
        AdaptiveWeight = Config.AdaptiveWeight 
        Weight_Normalization = Config.Weight_Normalization
        prob_norm_times = Config.prob_norm_times
        for i_iter in range(n_iters):
            init_loss = best_loss 
            if stable_count >= threshold_betwSquare and noise_weight > min_noise_weight:
                if not Config.fixed_Noiseweight:
                    noise_weight *= 0.98
                stable_count = 0
            p = self.p_selection(p_init, i_iter, n_iters, p_sche)  
            s = int(round(np.sqrt(p * n_patch_features / c)))
            s = min(max(s, 1), p_h)  
            self.square_size = s  
            print(f'No.square:{i_iter} square size:{self.square_size} noise_weight:{noise_weight}')
            if i_iter <= init_iters: 
                loc_h = np.random.randint(0, p_h - s + 1)
                loc_w = np.random.randint(0, p_w - s + 1)
                square_slice = (slice(0, c), slice(loc_h, loc_h + s), slice(loc_w, loc_w + s))
                print(f"initialize: randomly choose square slice: {(loc_h, loc_h + s, loc_w, loc_w + s)}")
            else:
                x_diff_map = self.Score.disp_diff_compute(patch_curr) 
                patch_prob_map = x_diff_map[(slice(p_t, p_t+p_h), slice(p_l, p_l+p_w))] # [p_h, p_w]
                for row in range(p_h):
                    for col in range(p_w):
                        row_index, col_index = row + p_t, col + p_l
                        row_top = row_index - math.floor(self.square_size / 2) if row_index - math.floor(self.square_size / 2) >= 0 else 0
                        row_bottom = row_index + math.ceil(self.square_size / 2) if row_index + math.ceil(self.square_size / 2) <= scene_H else scene_H
                        col_left = col_index - math.floor(self.square_size / 2) if col_index - math.floor(self.square_size / 2) >= 0 else 0
                        col_right = col_index + math.ceil(self.square_size / 2) if col_index + math.ceil(self.square_size / 2) <= scene_W else scene_W
                        sample_slice = (slice(row_top, row_bottom), slice(col_left, col_right))
                        patch_prob_map[row][col] = np.mean(x_diff_map[sample_slice])

                if np.max(patch_prob_map) > 0:
                    if prob_norm_times:
                        patch_prob_map = 3 * patch_prob_map / np.max(patch_prob_map)
                    else:
                        patch_prob_map =  patch_prob_map / np.max(patch_prob_map)
                patch_prob_map = softmax(patch_prob_map)

                idx = np.random.choice(np.arange(len(patch_prob_map.flatten())), p = patch_prob_map.flatten())
                idx = np.unravel_index(idx, patch_prob_map.shape)
                row_index, col_index = idx[0], idx[1]
                h_start = row_index - math.floor(self.square_size / 2) if row_index - math.floor(self.square_size / 2) >= 0 else 0
                h_end = row_index + math.ceil(self.square_size / 2) if row_index + math.ceil(self.square_size / 2) <= p_h else p_h
                w_start = col_index - math.floor(self.square_size / 2) if col_index - math.floor(self.square_size / 2) >= 0 else 0
                w_end = col_index + math.ceil(self.square_size / 2) if col_index + math.ceil(self.square_size / 2) <= p_w else p_w
                square_slice = (slice(0, c), slice(h_start, h_end), slice(w_start, w_end))
                print(f"targeted square: {h_start, h_end, w_start, w_end}")

            if Config.AdaptiveTrail:
                adapted_trail = 3 * self.square_size
                trail = adapted_trail if adapted_trail < init_trail else init_trail
            patch_square = patch_curr[square_slice]

            result_list = []
            steps_counter = 0
            for i in range(1, square_steps):  
                steps_counter += 1
                print('-' * 30)
                deltas = []
                avg_mean = []
                for j in range(num_pos):
                    if Config.noise_type == 'square':
                        noise = np.random.choice([-alpha, alpha], size=[trail, c, 1, 1])
                    elif Config.noise_type == 'discrete':  
                        noise = np.random.choice([0, 1], size=[trail, *patch_square.shape])  
                        noise = 2.0 * (noise - 0.5) * noise_weight  
                    else:
                        noise = np.random.rand(trail, *patch_square.shape)
                        noise = 2.0 * (noise - 0.5) * noise_weight
                    squares_candi = np.clip(noise + patch_square, 0, 1)  
                    noise = squares_candi - patch_square  
                    patches_candi = patch_curr.copy()
                    patches_candi = patches_candi[None, ...]  
                    patches_candi = np.repeat(patches_candi, trail, axis=0)
                    patches_candi[(slice(0, trail),) + square_slice] = squares_candi  

                    indice = torch.randperm(train_batch)[:1]
                    scene = scenes[indice]

                    scores = self.Score.get_gredient(patches_candi, patch_curr, scene, mask, new_standard_disp)

                    if Weight_Normalization: 
                        candi_y = normalize_score(scores.cpu().numpy())
                        if np.sum(np.abs(candi_y)) == 0:
                            candi_y = np.ones_like(candi_y)
                    else:
                        candi_y = scores.cpu().numpy()
                    mean_y = np.mean(candi_y)  
                    avg_mean.append(mean_y)

                    if mean_y == -1 or mean_y == 1:
                        delta = mean_y * np.mean(noise, axis=0)
                    else:
                        if minus_mean: 
                            delta = np.mean(noise * (candi_y - mean_y).reshape((trail, 1, 1, 1)), axis=0)
                        else:
                            if AdaptiveWeight == 'None':
                                delta = np.mean(noise * candi_y.reshape((trail, 1, 1, 1)), axis=0)
                            else:
                                pos_cnt = np.sum(candi_y > 0)
                                neg_cnt = trail - pos_cnt
                                if AdaptiveWeight == 'V1':
                                    candi_y[candi_y > 0] *= (2/(pos_cnt + eps))
                                    candi_y[candi_y <= 0] *= (2/(neg_cnt + eps))
                                elif AdaptiveWeight == 'V2':
                                    if int(pos_cnt) != 0 and int(neg_cnt) != 0:
                                        candi_y[candi_y > 0] *= ((2*neg_cnt)/trail)
                                        candi_y[candi_y <= 0] *= ((2*pos_cnt)/trail)
                                delta = np.mean(noise * candi_y.reshape((trail, 1, 1, 1)), axis=0)
                    
                    if np.isnan(delta).any() or np.isinf(delta).any():
                        print("NaN or inf")

                    if norm(delta) == 0:
                        print("norm == 0")
                    
                    delta = delta / norm(delta)
                    deltas.append(delta)
                delta = np.mean(deltas, axis=0)
                gradf = torch.from_numpy(delta)
                avg_mean = np.mean(avg_mean)
                
                gradf_flat = gradf.flatten()
                if i == 1:
                    grad_momentum = gradf
                    full_matrix   = torch.outer(gradf_flat, gradf_flat)
                else:
                    grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
                    full_matrix   = beta2 * full_matrix\
                                    + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)
                grad_momentum /= (1 - beta1 ** (i + 1))
                full_matrix   /= (1 - beta2 ** (i + 1))
                factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
                gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)

                patch_square = patch_square + lr * gradf.numpy() 
                patch_square = np.clip(patch_square, 0, 1)
                patch_curr[square_slice] = patch_square
                loss_curr = self.Score.score(patch_curr, mask, new_standard_disp)
                print("iter", i, "score_curr", loss_curr)

                if loss_curr < best_loss:
                    best_loss = loss_curr
                    patch_best[:] = patch_curr
                    stable_count_inSquare = 0
                else:
                    stable_count_inSquare += Config.gap

                if stable_count_inSquare >= threshold_inSquare:
                    stable_count_inSquare = 0
                    break

            if init_loss > best_loss:
                stable_count = 0
            else:
                stable_count += 1

            dis, env = self.Score.score(patch_curr, mask, new_standard_disp, epe=True)
            result_list.append((loss_curr, dis, env))
            print(f'train dis: {dis}, train env: {env}')

        loss_curr = self.eval_Score.score(patch_best, mask, new_standard_disp)
        print(f'Loss: {loss_curr.item()}')
        eval_dis, eval_env = self.eval_Score.score(patch_best, mask, new_standard_disp, epe=True)
        print(f'evaluation dis: {eval_dis}, evaluation env: {eval_env}')

        file_name = './logs/{}/result.txt'.format(test_name)
        with open(file_name, 'w') as f:
            for i, (loss, dis, env) in enumerate(result_list):
                f.write(f'epoch {i}: loss={loss}, dis={dis}, env={env}\n')
            f.write(f'-----------------------------------------------\n')
            f.write(f'test: loss={loss_curr}, eval_dis={eval_dis}, eval_env={eval_env}\n')

        best_patch = torch.Tensor(patch_best)
        patch_path = './logs/{}'.format(test_name)
        self.eval_Score.viz_test(best_patch, patch_path, 2)
        scene_image = self.eval_Score.viz(best_patch)
        scene_image.save('./logs/{}/image_scene.png'.format(test_name))
        patch_best1 = np.transpose(patch_best, (1, 2, 0))
        patch_image = Image.fromarray((patch_best1 * 255).astype(np.uint8))
        patch_image.save('./logs/{}/image_patch.png'.format(test_name))
        torch.save(best_patch, './logs/{}/best_patch.pt'.format(test_name))

    def sparse_RS(self, n_iters, p_init, p_sche='v1', test_name = '1'):        
        c = 3
        _, _, p_h, p_w = self.patch_area
        n_patch_features = c * p_h * p_w
        s = int(round(np.sqrt(p_init * n_patch_features / c)))
        s = min(max(s, 1), p_h)  
        self.square_size = s
        patch_curr = np.full((c, p_h, p_w), 0.) 
        for i in range(1000):
            init_s = np.random.randint(1, min(p_h, p_w))
            loc_h = np.random.randint(0, p_h - init_s + 1)
            loc_w = np.random.randint(0, p_w - init_s + 1)
            square_slice = (slice(0, c), slice(loc_h, loc_h + init_s), slice(loc_w, loc_w + init_s))
            patch_curr[square_slice] = np.random.choice([0., 1.], size=[c, 1, 1])
        patch_curr = np.clip(patch_curr, 0., 1.)
        patch_best = patch_curr.copy()

        scene_loader_iter = iter(self.train_loader)  
        scenes = next(scene_loader_iter)  
        train_batch, c, scene_H, scene_W = scenes.shape  

        col_index = 400
        scene_standard = scenes[0:1]      
        standard_disp = self.model(scene_standard.to(Config.device))
        depth_column = standard_disp[:, :, :, col_index:col_index+1]
        new_standard_disp = depth_column.expand(-1, -1, -1, standard_disp.size(3))

        m_t, m_l, m_h, m_w = get_mask_area(self.patch_area)
        mask = torch.zeros((1, 1, scene_H, scene_W)).to(Config.device)
        mask[:, :, m_t : m_t + m_h, m_l : m_l + m_w] = 1  

        best_loss = self.Score.score(patch_curr, mask, new_standard_disp)
        result_list = []
        cn_ft_idx = None 
        for i_iter in range(n_iters):
            patch_curr[:] = patch_best
            p = self.p_selection(p_init, i_iter, n_iters, p_sche)
            s = int(round(np.sqrt(p * n_patch_features / c)))
            s = min(max(s, 1), p_h)
            self.square_size = s
            loc_h = np.random.randint(0, p_h - s + 1)
            loc_w = np.random.randint(0, p_w - s + 1)
            if s == 1 and cn_ft_idx is None:
                cn_ft_idx = i_iter + (n_iters - i_iter) // 2
            if cn_ft_idx is not None and i_iter > cn_ft_idx:
                loc_channel = np.random.randint(0, 3) 
                square_slice = (slice(loc_channel, loc_channel + 1), slice(loc_h, loc_h + s), slice(loc_w, loc_w + s))
            else:
                square_slice = (slice(0, c), slice(loc_h, loc_h + s), slice(loc_w, loc_w + s))

            while np.sum(np.abs(np.clip(patch_curr[square_slice], 0, 1) - patch_best[square_slice]) < 10**-7) == patch_best[square_slice].size:
                if cn_ft_idx is not None and i_iter > cn_ft_idx:
                    patch_curr[square_slice] = 1 - patch_curr[square_slice]
                else:
                    patch_curr[square_slice] += np.random.choice([-1., 1.], size=[c, 1, 1])
                patch_curr = np.clip(patch_curr, 0., 1.)
            curr_loss = self.Score.score(patch_curr, mask, new_standard_disp)
            result_list.append(curr_loss)
            print("-" * 30)
            if curr_loss < best_loss:
                patch_best[:] = patch_curr
                best_loss = curr_loss

        eval_dis, eval_env = self.eval_Score.score(patch_best, mask, new_standard_disp, epe=True)
        print(f'evaluation dis: {eval_dis}, evaluation env: {eval_env}')

        file_name = './logs/{}/result.txt'.format(test_name)
        with open(file_name, 'w') as f:
            for i, (loss, dis, env) in enumerate(result_list):
                f.write(f'epoch {i}: loss={loss}, dis={dis}, env={env}\n')
            f.write(f'-----------------------------------------------\n')
            f.write(f'test: eval_dis={eval_dis}, eval_env={eval_env}\n')

        best_patch = torch.Tensor(patch_best)
        patch_path = './logs/{}'.format(test_name)
        self.eval_Score.viz_test(best_patch, patch_path, 2)
        scene_image = self.eval_Score.viz(best_patch)
        scene_image.save('./logs/{}/image_scene.png'.format(test_name))
        patch_best = np.transpose(patch_best, (1, 2, 0))
        patch_image = Image.fromarray((patch_best * 255).astype(np.uint8))
        patch_image.save('./logs/{}/image_patch.png'.format(test_name))
        torch.save(best_patch, './logs/{}/best_patch.pt'.format(test_name))      

    def hardbeat_attack(self, total_steps=100, K=4, num_pos=100, num_init=10, trail=30, test_name = '1'):
        scene_loader_iter = iter(self.train_loader)
        scenes = next(scene_loader_iter)
        c = scenes.shape[1]
        train_batch, c, scene_H, scene_W = scenes.shape  
        scene_size = (scene_H, scene_W)
        p_t, p_l, p_h, p_w = self.patch_area
        patch_curr = np.random.rand(c, p_h, p_w)

        col_index = 400
        scene_standard = scenes[0:1]    
        standard_disp = self.model(scene_standard.to(Config.device))
        depth_column = standard_disp[:, :, :, col_index:col_index+1]
        new_standard_disp = depth_column.expand(-1, -1, -1, standard_disp.size(3))

        m_t, m_l, m_h, m_w = get_mask_area(self.patch_area)
        mask = torch.zeros((1, 1, scene_H, scene_W)).to(Config.device)
        mask[:, :, m_t : m_t + m_h, m_l : m_l + m_w] = 1  

        patch_best = patch_curr.copy()
        score_best = self.Score.score(patch_best, mask, new_standard_disp)
        
        for i in range(num_init):
            patch_curr = np.random.rand(c, p_h, p_w)
            score = self.Score.score(patch_best, mask, new_standard_disp)
            if score < score_best:
                score_best = score
                patch_best[:] = patch_curr
                print(f"Initialize: step: {i} best score: {score_best}")

        hist_patch = [patch_best]
        hist_score = np.array([1 / score_best])
        sim_graph = np.eye(100)
        last_mean_y = np.array([0] * num_pos)
        beta1 = Config.beta1
        beta2 = Config.beta1
        eps = Config.eps
        lr = Config.lr
        early_stop_step = 0
        result_list = []
        for i in range(1, total_steps):
            if not Config.hardbeat_oneway:
                if len(hist_score) > K:
                    topk_idx = np.argpartition(hist_score, -K)[-K:]
                else:
                    topk_idx = np.arange(len(hist_score))
                topk_prob = softmax(hist_score[topk_idx])
                curr_idx = np.random.choice(topk_idx, size=1, p=topk_prob)[0]
                u = np.random.rand(1)[0]
                if u <= min(1, hist_score[curr_idx] / hist_score[-1]):
                    patch_curr = hist_patch[curr_idx]
                    if u <= 0.5 and i > 2:
                        neighbor_idx = find_neighbor(sim_graph, curr_idx, hist_score)
                        neighbor_patch = hist_patch[neighbor_idx]
                        alpha = np.random.rand(1)[0]
                        patch_curr = alpha * patch_curr + (1-alpha) * neighbor_patch
                else:
                    patch_curr = hist_patch[-1]
                    curr_idx = len(hist_patch) - 1

            deltas = []
            avg_mean = []
            for j in range(num_pos): 
                noise = np.random.rand(trail, *patch_curr.shape)
                noise = 2.0 * (noise - 0.5) * 0.5
                patches_candi = np.clip(noise + patch_curr, 0, 1)
                noise = patches_candi - patch_curr

                indice = torch.randperm(scenes.shape[0])[:1]
                scene = scenes[indice]

                scores = self.Score.get_gredient(patches_candi, patch_curr, scene, mask, new_standard_disp)

                candi_y = scores.cpu().numpy()
                candi_y = np.sign(candi_y)
                mean_y = np.mean(candi_y)  
                avg_mean.append(mean_y)
                diff_y = mean_y - last_mean_y[j]
                if diff_y > 0:
                    diff_y = np.exp(diff_y)
                    if mean_y >= 1:
                        diff_y /= 5
                else:
                    diff_y = np.log(diff_y + 3)
                if mean_y == -1 or mean_y == 1:
                    delta = mean_y * np.mean(noise, axis=0)
                else:
                    delta = np.mean(noise * (candi_y - mean_y).reshape((trail, 1, 1, 1)), axis=0)
                delta = delta / norm(delta)
                last_mean_y[j] = mean_y
                deltas.append(delta)
            gradf = torch.from_numpy(np.mean(deltas, axis=0))
            avg_mean = np.mean(avg_mean)

            gradf_flat = gradf.flatten()
            if i == 1:
                grad_momentum = gradf
                full_matrix   = torch.outer(gradf_flat, gradf_flat)
            else:
                grad_momentum = beta1 * grad_momentum + (1 - beta1) * gradf
                full_matrix   = beta2 * full_matrix\
                                + (1 - beta2) * torch.outer(gradf_flat, gradf_flat)
            grad_momentum /= (1 - beta1 ** (i + 1))
            full_matrix   /= (1 - beta2 ** (i + 1))
            factor = 1 / torch.sqrt(eps + torch.diagonal(full_matrix))
            gradf = (factor * grad_momentum.flatten()).reshape_as(gradf)
            
            patch_curr = patch_curr + lr * gradf.numpy()
            patch_curr = np.clip(patch_curr, 0, 1)

            if not Config.hardbeat_oneway:
                for t in range(len(hist_patch)):
                    p1 = hist_patch[t].flatten()
                    p2 = patch_curr.flatten()
                    sim = ((p1.dot(p2) / (norm(p1) * norm(p2))) + 1) / 2
                    sim_graph[i % 100, t] = sim
                    sim_graph[t, i % 100] = sim
            
            score_curr = self.Score.score(patch_curr, mask, new_standard_disp)

            if not Config.hardbeat_oneway:
                hist_patch.append(patch_curr)
                hist_score = np.append(hist_score, 1 / score_curr)
                if len(hist_patch) == 100:
                    hist_patch.pop(0)
                    hist_score = np.delete(hist_score, 0)
                    sim_graph = np.delete(sim_graph, 0, axis=0)
                    sim_graph = np.delete(sim_graph, 0, axis=1)
                    new_row = np.zeros((1, sim_graph.shape[1]))
                    sim_graph = np.vstack((sim_graph, new_row))
                    new_colomn = np.zeros((sim_graph.shape[0], 1))
                    sim_graph = np.hstack((sim_graph, new_colomn))
                    sim_graph[-1][-1] = 1

            if score_curr < score_best:
                score_best = score_curr
                patch_best[:] = patch_curr
                early_stop_step = 0
            else:
                early_stop_step += 1
            print(f'Epoch {i}, Loss: {score_curr.item()}')

            dis, env = self.Score.score(patch_curr, mask, new_standard_disp, epe=True)
            result_list.append((loss_curr, dis, env))
            print(f'train dis: {dis}, train env: {env}')
            if early_stop_step >= 5:
                print('early stop!')
                break
            
        loss_curr = self.eval_Score.score(patch_best, mask, new_standard_disp)
        print(f'Loss: {loss_curr.item()}')
        eval_dis, eval_env = self.eval_Score.score(patch_best, mask, new_standard_disp, epe=True)
        print(f'evaluation dis: {eval_dis}, evaluation env: {eval_env}')
        file_name = './logs/{}/result.txt'.format(test_name)
        with open(file_name, 'w') as f:
            for i, (loss, dis, env) in enumerate(result_list):
                f.write(f'epoch {i}: loss={loss}, dis={dis}, env={env}\n')
            f.write(f'-----------------------------------------------\n')
            f.write(f'test: loss={loss_curr}, eval_dis={eval_dis}, eval_env={eval_env}\n')

        best_patch = torch.Tensor(patch_best)
        patch_path = './logs/{}'.format(test_name)
        self.eval_Score.viz_test(best_patch, patch_path, 2)
        scene_image = self.eval_Score.viz(best_patch)
        scene_image.save('./logs/{}/image_scene.png'.format(test_name))
        patch_best = np.transpose(patch_best, (1, 2, 0))
        patch_image = Image.fromarray((patch_best * 255).astype(np.uint8))
        patch_image.save('./logs/{}/image_patch.png'.format(test_name))
        torch.save(best_patch, './logs/{}/best_patch.pt'.format(test_name))
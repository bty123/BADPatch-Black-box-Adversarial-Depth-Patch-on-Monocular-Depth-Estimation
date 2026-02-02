import torch

class Config(object):
    square_batch_size = 5
    device = torch.device("cuda")
    kitti_dataset_root = "dataset/KITTI/"
    Carla_dataset_root = "dataset/Carla/"
    Carla_own_dataset_root = "dataset/Carla_own/"
    log_dir = "logs"
    input_W = 1024
    input_H = 320
    input_W_DepthAnyting = 1022
    input_H_DepthAnyting = 322
    train_scenes = 20
    scenes_batch = 5
    eps = 1e-8
    init_noise_weight = 0.1
    min_noise_weight = 0.03
    lr = 0.1 
    white_lr = 0.01
    beta1 = 0.5 
    beta2 = 0.5 
    gap = 1 
    AdaptiveTrail = False
    fixed_Noiseweight = False
    topk = False 
    minus_mean = False 
    noise_type = 'discrete'  
    prob_norm_times = False
    AdaptiveWeight = 'V1' 
    Weight_Normalization = True 
    UseAdam = True
    Oneway = True 
    hardbeat_oneway = True
    model_scene_sizes_WH = {
        'monodepth2': (input_H, input_W),
        'depthhints': (input_H, input_W),
        'lite_mono': (input_H, input_W),
        'SQLdepth': (input_H, input_W),
        'MiDaS': (input_H, input_W),
        'DepthAnything': (input_H_DepthAnyting, input_W_DepthAnyting)
    }
    threshold_betwSquare = {
        'monodepth2': 5, 
        'depthhints': 5,
        'lite_mono': 5,
        'SQLdepth':5,
        'DepthAnything': 5,
        'MiDaS': 5
    }
    threshold_inSquare = {
        'monodepth2': 5, 
        'depthhints': 5,
        'lite_mono': 5,
        'SQLdepth':5,
        'DepthAnything': 5,
        'MiDaS': 5
    }
    

import os
import options
import torch
from attack.depth_model import import_depth_model
from attack.Attack_patch import Attack_patch
from my_utils import get_patch_area
from config import Config
import time


def main():
    args = options.parse()
    log_dir =  os.path.join(Config.log_dir, args.test_name)
    os.makedirs(log_dir, exist_ok=True)
    model_name  = args.model_name
    scene_size  = Config.model_scene_sizes_WH[model_name]
    patch_area = get_patch_area(scene_size)
    model = import_depth_model(model_name).to(Config.device).eval()        
    
    attack_patch = Attack_patch(model ,model_name, patch_area, args.n_batch, args.batch_size, args.trail)
    if args.attack_method == 'whitebox':
            attack_patch.whitebox_attack(test_name = args.test_name)
    with torch.no_grad():
        if args.attack_method == 'blackbox':
            attack_patch.blackbox_attack(trail=args.trail, test_name = args.test_name)
            
        elif args.attack_method == 'badPart':
            attack_patch.BadPart(alpha=args.alpha, n_iters=args.n_iter, init_iters=args.init_iters, p_init=args.p_init, p_sche=args.p_sche,
                                    square_steps=args.square_steps, num_pos=args.num_pos, trail=args.trail, test_name = args.test_name)

        elif args.attack_method == 'sparse_RS':
            attack_patch.sparse_RS(n_iters=args.n_iter, p_init=args.p_init, p_sche=args.p_sche, test_name = args.test_name)

        elif args.attack_method == 'hardbeat':
            attack_patch.hardbeat_attack(test_name = args.test_name)
        else:
            raise RuntimeError(f'The attack method {args.attack_method} is not supported!')

if __name__ == "__main__":
    t = time.time()
    main()
    print(f'time:{time.time() - t:.4f}s')
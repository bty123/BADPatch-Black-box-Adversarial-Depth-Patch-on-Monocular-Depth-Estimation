import argparse

def parse():
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model_name', type=str, default='monodepth2', choices=['monodepth2', 'depthhints', 'DepthAnything', 'SQLdepth', 'MiDaS', 'lite_mono'], required=False, help='name of the subject model.')
    parser.add_argument('--attack_method', type=str, default='blackbox', choices=['blackbox', 'whitebox', 'badPart', 'sparse_RS', 'hardbeat'], required=False, help='name of the attack method.')
    parser.add_argument('--n_iter', type=int, default=200, help='maximim iterations to try')
    parser.add_argument('--alpha', type=float, default=0.1,  help='step of each attack')
    parser.add_argument('--patch_ratio', type=float, default=0.02,  help='Proportion of patch area')
    parser.add_argument('--aspect_ratio', type=float, default=2,  help='Proportion of weight and height')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size of kitti loader')
    parser.add_argument('--n_batch', type=int, default=10, help='number of batch to evaluate per iteration')
    parser.add_argument('--trail', type=int, default=25, help='number of noise sampled')
    parser.add_argument('--num_pos', type=int, default=1, help='number of different position, if 1 then fixed position')
    parser.add_argument('--p_init', type=float, default=0.025, help='initial ratio of square patch')
    parser.add_argument('--init_iters', type=int, default=100, help='number of initial iters of the attack')
    parser.add_argument('--square_steps', type=int, default=200, help='number of maximum iters of the attack in a square area')
    parser.add_argument('--p_sche', type=str, default='v6', help='square size changing schdule version')
    parser.add_argument('--log_dir', type=str, help='log dir', required=False)
    parser.add_argument('--test_name', type=str, default='monodepth2/1', required=False, help='name for this test')
    args = parser.parse_args()
    return args

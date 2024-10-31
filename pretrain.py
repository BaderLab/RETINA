"""
Contains the code to set up the pre-training process.
"""
import argparse
import logging
import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from networks.vit_seg_pretrainmodeling import VisionTransformer as ViT_seg
from networks.vit_seg_pretrainmodeling import CONFIGS as CONFIGS_ViT_seg
from pretrainer import trainer_synapse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--max_epochs', type=int,
                    default=5000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=256, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=8, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=4,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.003,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--current_device', default=0, type=int, help='')
parser.add_argument('--trained_step', default=0, type=int, help='')
args = parser.parse_args()
def set_deterministic(seed=42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    set_deterministic(args.seed)
    print("availalbecuda:", torch.cuda.is_available())
    ngpus_per_node = torch.cuda.device_count()
    print("ngpus_per_node:", ngpus_per_node)
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank 
    current_device = local_rank
    torch.cuda.set_device(current_device)
    current_device = torch.cuda.current_device()
    args.current_device = current_device
    print("current:", current_device)
    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    
    print("process group ready!")
    print('From Rank: {}, ==> Making model..'.format(rank))
    dataset_name = args.dataset
    args.is_pretrain = False
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_skip = args.n_skip
    # the default parameters from the transunet paper
    config_vit.hidden_size = 768
    config_vit.transformer.num_heads = 12
    config_vit.transformer.mlp_dim = 3072
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    print("heads:", config_vit.transformer.num_heads)
    print("layers:", config_vit.transformer.num_layers)
    # This section is for loading the pre-trained parameters
    # from the ImageNet pre-training
    #pretraining = 'imagenet21k_R50+ViT-B_16.npz'
    #print("has pretrained with the imagenet")
    #net.load_from(weights=np.load(pretraining))
    #print(f'Successfully loaded parameters from {pretraining}')
    trained_step = args.trained_step
    if trained_step == 0:
        print("Pretrain begins from the start.")
    else:
        # add your trained file path here.
        path = ""
        print(f'have trained {trained_step} steps')
        state_dict = torch.load(path, map_location=torch.device('cuda'))
        net.load_state_dict(state_dict, strict=True)
        
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Using model with {params} trainable parameters!')
    args.base_lr = 0.003
    args.noise_range = (400,1200)
    args.max_holes = 32
    print("gaussian noise range:", args.noise_range)
    print("max and min number of masked holes:", args.max_holes)
    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, local_rank)
    print("All done!")
"""
This file is to set up the fine-tuning process after pre-training.
train.py can be run by the slurm file in each benchmark directory.
Also, for each benchmark, the configuration files are also included 
in each corresponding directory, which is very easy to use.

To improve the computation speed, the distribution of multiple GPUs 
are used and one could change the number of GPU easily based on the 
resource and the datasets.

We took the reference on TransUNet model:
(Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., et al. (2021).)
"""
import argparse
import logging
import os, yaml
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_synapse
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='/RETINA/cremi/cremi.yaml')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum iteration number to train')
parser.add_argument('--max_epochs', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=0,
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
# world size should be the GPU number
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--current_device', default=0, type=int, help='')
args = parser.parse_args()
def set_deterministic(seed=42):
    # Ensure deterministic behavior
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
    print("seed:", args.seed)
    args.ngpus = torch.cuda.device_count()
    print("ngpus_per_node:", args.ngpus)
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    print("local_rank:", local_rank)
    rank = int(os.environ.get("SLURM_NODEID"))*args.ngpus + local_rank 
    current_device = local_rank
    torch.cuda.set_device(current_device)
    current_device = torch.cuda.current_device()
    args.current_device = current_device
    print("current:", current_device)
    print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
    #init the process group
    print("world_size:", args.world_size)
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    
    print("process group ready!")
    print('From Rank: {}, ==> Making model..'.format(rank))
    dataset_name = args.dataset
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    args.root_path = config['data_dir']
    if config.get('valid_dir') != None:
        args.valid_path = config['valid_dir']
    else:
        args.valid_path = None
    args.num_classes = config['num_classes']
    args.class_names = config['class_names']
    args.batch_size = config['bsz']
    args.base_lr = config['lr']
    args.mean = config['norms']['mean']
    args.std = config['norms']['std']
    args.augmentations = config['augmentations']
    args.num_prints = config['num_prints']
    args.max_iterations = config['iters']
    args.save_interval = config['save_interval']
    args.model_dir = config['model_dir']
    print("model_dir:", args.model_dir)
    args.finetuned = 0
    if 'train_length' not in config:
        args.train_length = 1
    else:
        args.train_length = config['train_length'] # determine how many percent of total train images are used.
    
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.transformer.num_layers = config['transformer_layer']
    config_vit.n_classes = config['num_classes']
    config_vit.n_skip = args.n_skip
    config_vit.mlp_ratio = 4
    
    pretrain = config['pretrain']
    if pretrain == 'imagenet' or pretrain == 'incem':
        # The default set of parameters same as the transunet paper;
        # To accommodate the pretrained imagenet pretrained model.
        config_vit.hidden_size = 768
        config_vit.transformer.num_heads = 12
        config_vit.transformer.mlp_dim = 3072
    
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    print("vithiddensize", config_vit.hidden_size)
    print("mlp", config_vit.transformer.mlp_dim)
    print("heads:", config_vit.transformer.num_heads)
    print("layers:", config_vit.transformer.num_layers)
    print("dropout:", config_vit.transformer.dropout_rate)
    print("attentiondropout:", config_vit.transformer.attention_dropout_rate)
    print("lr", args.base_lr)
    print("data_dir:", config['data_dir'])
    print(f"take {args.train_length * 100}% of the train images as the training images")
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    
    if pretrain == True or pretrain == 'incem':
        pretraining = config['pretrain_file']
        print("finetune with the normal transunet pretrained model")
        state_dict = torch.load(pretraining, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if 'segmentation_head' not in k and 'decoder' not in k} 
        msg = net.load_state_dict(state_dict, strict=False)
        print(f'Successfully loaded parameters from {pretraining}')
        num = 0
        for name, param in net.named_parameters():
            if name.startswith('transformer.encoder'):
                param.requires_grad = False
                num = num + 1
            else:
                param.requires_grad = True
        print("only freeze the transformer layers")
        print("num:", num)
        
    elif pretrain == 'imagenet':
        pretraining = config['pretrain_file']
        print("finetune with the imagenet")
        net.load_from(weights=np.load(pretraining))
        print(f'Successfully loaded parameters from {pretraining}')
        num = 0
        for name, param in net.named_parameters():
            if name.startswith('transformer.encoder'):
                param.requires_grad = False
                num = num + 1
            else:
                param.requires_grad = True
        print("only freeze the transformer layers")
        print("num:", num)
    else:
        print("no pretrain, no freeze")
        for name, param in net.named_parameters():
            param.requires_grad = True
    
    # Here is for contine the finetune
    if config.get('finetuned') != None:
        finetuned = config['finetuned']
    else:
        finetuned = False
        
    if finetuned == True:
        finetuned_file = config['finetuned_file']
        parts = finetuned_file.split('/')
        filename = parts[-1]
        iteration_number = int(filename.split('_')[-1].split('.')[0])
        state_dict = torch.load(finetuned_file, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        msg = net.load_state_dict(new_state_dict, strict=True)
        print(f'Successfully loaded parameters from {finetuned_file}!')
        print(f'have finetuned {iteration_number} steps!')
        args.finetuned = iteration_number

    # the weight for the multiclass benchmark dataset.
    if config.get('class_weights') != None:
        args.class_weights = config['class_weights']
        print("class weights:", args.class_weights)
    else:
        args.class_weights = None
    
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Using model with {params} trainable parameters!')
    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, local_rank, args.train_length)

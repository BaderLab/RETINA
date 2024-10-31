"""
Contains the code after fine-tuning with the fine-tuned model parameters.
Referenced from: Conrad, R. and Narayan, K. (2021). 
"""

import numpy as np
import os, argparse, cv2, yaml
import torch
import torch.nn as nn
import SimpleITK as sitk
from glob import glob

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from monai.inferers import sliding_window_inference
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
 
def factor_pad_tensor(tensor, factor=32):
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    return nn.ReflectionPad2d((0, pad_right, 0, pad_bottom))(tensor)

def parse_args():
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Run orthoplane inference on volumes in a given directory')
    parser.add_argument('--state_path', default='', type=str, metavar='state_path', help='Path to model state file')
    parser.add_argument('--mode3d', dest='mode3d', type=str, metavar='mode3d', choices=['orthoplane', 'stack'], default='stack',
                        help='Inference mode. Choice of orthoplane or stack.')
    parser.add_argument('--threshold3d', type=float, metavar='threshold3d', help='Prediction confidence threshold [0-1]')
    parser.add_argument('--eval_classes3d', dest='eval_classes3d', type=int, metavar='eval_classes', nargs='+',
                        help='Index/indices of classes to evaluate for multiclass segmentation')
    parser.add_argument('--mask_prediction3d', action='store_true', help='whether to evaluate IoU by first masking with ground truth')
    
    #Cheng changed here
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
    parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum iteration number to train')
    parser.add_argument('--max_epochs', type=int,
                    default=10000, help='maximum epoch number to train')
    parser.add_argument('--n_gpu', type=int, default=4, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=0,
                    help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
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
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    args = parse_args()
    with open(args['config'], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    #add the state path to config
    config['state_path'] = args['state_path']
    #read in the arguments
    test_dir = config['test_dir3d']
    state_path = config['state_path']
    save_dir = config['save_dir3d']
    mode = config['mode3d']
    threshold = config['threshold3d']

    if threshold is None:
        threshold = 0.5
    
    eval_classes = config['eval_classes3d']
    mask_prediction = config['mask_prediction3d']
    
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            print(f'Created directory {save_dir}')     

    impaths = glob(os.path.join(test_dir, 'images/*'))
    
    inference_only = True
    if os.path.isdir(os.path.join(test_dir, 'masks')):
        inference_only = False
    
    state = torch.load(state_path, map_location='cpu')

    if 'run_id' in state:
        run_id = state['run_id']
    else:
        run_id = None
    
    norms = config['norms']

    gray_channels = 1
    num_classes = config['num_classes']

    eval_tfs = Compose([
        Normalize(mean=norms['mean'], std=norms['std']),
        ToTensorV2()
    ])
    config_vit = CONFIGS_ViT_seg[args['vit_name']]
    config_vit.n_classes = num_classes
    config_vit.n_skip = args['n_skip']
    config_vit.mlp_ratio = 4
    config_vit.transformer.num_layers = config['transformer_layer']
    
    if config['pretrain'] == 'imagenet' or config['pretrain'] == 'incem':

        print(config['pretrain'])
        config_vit.hidden_size = 768
        config_vit.transformer.num_heads = 12
        config_vit.transformer.mlp_dim = 3072
    
    if args['vit_name'].find('R50') != -1:
        config_vit.patches.grid = (int(args['img_size'] / args['vit_patches_size']), int(args['img_size'] / args['vit_patches_size']))
    config_vit = CONFIGS_ViT_seg[args['vit_name']]
    model = ViT_seg(config_vit, img_size=args['img_size'], num_classes=config_vit.n_classes)
    state_new = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state_new)

    model = model.cuda()

    model.eval()
    
    if mode == 'stack':
        axes = [0]
    elif mode == 'orthoplane':
        axes = [0, 1, 2]
    else:
        raise Exception(f'Inference mode must be orthoplane or stack, got {mode}')
    
    scaling = 255 / len(axes)
    

    threshold = int(255 * threshold)
    

    volume_mean_ious = []
    for imvol in impaths:
        print(f'Loading {imvol}')
        orig_vol = sitk.ReadImage(imvol)


        im_vol = sitk.GetArrayFromImage(orig_vol)
        print(f'Volume size {im_vol.shape}')

        #assert that the volume type is uint8
        assert(im_vol.dtype == np.uint8), \
        'Image volume must have 8-bit unsigned voxels!'

        prediction_volume = np.zeros((num_classes, *im_vol.shape), dtype=np.uint8) #e.g. (3, 256, 256, 256)

        for ax in axes:
            print(f'Predicting over axis {ax}')
            stack = np.split(im_vol, im_vol.shape[ax], axis=ax)
            for index, image in enumerate(stack):
                if gray_channels == 3:
                    image = cv2.cvtColor(np.squeeze(image), cv2.COLOR_GRAY2RGB)
                else:
                    #add an empty channel dim
                    image = np.squeeze(image)[..., None]

                image = eval_tfs(image=image)['image'].unsqueeze(0)

                #load image to gpu
                image = image.cuda()

                #get the image size and pad the image to a factor
                h, w = image.size()[2:]

                with torch.no_grad():
                    prediction = sliding_window_inference(image, (224, 224), 4, model, overlap=0.7)

                    if num_classes == 1:
                        prediction = nn.Sigmoid()(prediction) #(1, 1, H, W)
                    else:
                        prediction = nn.Softmax(dim=1)(prediction) #(1, NC, H, W)
                    prediction = (prediction.squeeze(0).detach().cpu().numpy() * scaling).astype(np.uint8) #(NC, H, W)

                    if ax == 0:
                        prediction_volume[:, index] += prediction
                    elif ax == 1:
                        prediction_volume[:, :, index] += prediction
                    else:
                        prediction_volume[:, :, :, index] += prediction
            
        if num_classes == 1:
            prediction_volume = (prediction_volume > threshold).astype(np.uint8)[0] #(1, D, H, W) --> (D, H, W)
        else:
            prediction_volume = np.argmax(prediction_volume, axis=0).astype(np.uint8) #(NUM_CLASSES, D, H, W) --> (D, H, W)

        if save_dir is not None:
            #convert the numpy volume to a SimpleITK image
            save_prediction = sitk.GetImageFromArray(prediction_volume)

            save_prediction.CopyInformation(orig_vol)
            
            #get the volume name from it's path
            vol_name = imvol.split('/')[-1]

            sitk.WriteImage(save_prediction, os.path.join(save_dir, vol_name))
            del save_prediction

        if not inference_only:
            vol_name = imvol.split('/')[-1]
            gtvol = os.path.join(test_dir, f'masks/{vol_name}')
            
            if not os.path.isfile(gtvol):
                print(f'No ground truth found at {gtvol} for {imvol}')
            else:
                gtvolume = sitk.GetArrayFromImage(sitk.ReadImage(gtvol))
                gtvolume = gtvolume.astype(np.uint8)

                #in some cases, the entire gtvolume may not be labeled
                #this is the case for the Guay benchmark. in that case
                #we need to ignore any predictions that fall outside
                #the labeled region in the gtvolume
                if mask_prediction:
                    mask = gtvolume > 0
                    prediction_volume *= mask

                #if we defined eval classes, then we'll only
                #consider them during evaluation. otherwise we'll
                #consider all labels
                if eval_classes is None:
                    if num_classes == 1:
                        eval_classes = [1]
                    else:
                        eval_classes = list(range(1, num_classes))

                #loop over each of the eval_classes and
                #calculate the IoUs
                class_ious = []
                for label in eval_classes:
                    label_pred = prediction_volume == label
                    label_gt = gtvolume == label
                    intersect = np.logical_and(prediction_volume == label, gtvolume == label).sum()
                    union = np.logical_or(prediction_volume == label, gtvolume == label).sum()

                    #add small epsilon to prevent zero division
                    iou = (intersect + 1e-5) / (union + 1e-5)

                    #print the class IoU
                    print(f'Class {label} IoU 3d: {iou}')

                    #store the result
                    class_ious.append(intersect / union)

                #calculate and print the mean iou
                mean_iou = np.mean(class_ious).item()
                print(f'Mean IoU 3d: {mean_iou}')
                
            volume_mean_ious.append(mean_iou)
   
    if not inference_only:
        mean_iou = np.mean(volume_mean_ious).item()
        print(f'Overall mean IoU 3d: {mean_iou}')

"""
Contains the code after fine-tuning with the fine-tuned model parameters.
Referenced from: Conrad, R. and Narayan, K. (2021). 
"""
import os, sys, argparse, warnings, cv2, yaml
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from skimage import measure
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from copy import deepcopy
from datasets.dataset_synapse import SegmentationData
from PIL import Image

from monai.inferers import sliding_window_inference
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def mean_iou(output, target):
    if target.ndim == output.ndim - 1:
        target = target.unsqueeze(1)
    n_classes = output.size(1)
    empty_dims = (1,) * (target.ndim - 2)

    if n_classes > 1:
        k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
        target = (target == k)

        output = nn.Softmax(dim=1)(output)
    else:
        #just sigmoid the output
        output = (nn.Sigmoid()(output) > 0.5).long()

    #cast target to the correct type for operations
    target = target.type(output.dtype)
    dims = (0,) + tuple(range(2, target.ndim))
    intersect = torch.sum(output * target, dims)

    #compute the union, (N,)
    union = torch.sum(output + target, dims) - intersect

    #avoid division errors by adding a small epsilon
    iou = (intersect + 1e-7) / (union + 1e-7)

    return iou.mean().item()

def parse_args(): 
    #setup the argument parser
    parser = argparse.ArgumentParser(description='Evaluate on set of 2d images')
    parser.add_argument('--config', default='/RETINA/perez_nuclei/perez_nuclei.yaml', type=str, metavar='config', help='Path to a config yaml file')
    parser.add_argument('--state_path', default='', type=str, metavar='state_path', help='Path to model state file')
    parser.add_argument('--threshold2d', type=float, metavar='threshold2d', help='Prediction confidence threshold [0-1]')
    parser.add_argument('--eval_classes2d', dest='eval_classes2d', type=int, metavar='eval_classes', nargs='+',
                        help='Index/indices of classes to evaluate for multiclass segmentation')
    parser.add_argument('--instance_match2d', action='store_true', help='whether to evaluate IoU by instance matching')
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
    config['state_path'] = args['state_path']
    test_dir = config['test_dir2d']
    state_path = config['state_path']
    save_dir = config['save_dir2d']
    threshold = None
    
    if threshold is None:
        threshold = 0.5
    eval_classes = config['eval_classes2d']
    if config['experiment_name'][:5] == 'perez':
        
        instance_match = config['instance_match2d']
    else:
        instance_match = None
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            print(f'Created directory {save_dir}')
            
    state_path = config['state_path']
    state = torch.load(state_path, map_location='cpu')
    if 'run_id' in state:
        run_id = state['run_id']
    else:
        run_id = None
    
    norms = config['norms']
    print(norms)
    gray_channels = 1
    num_classes = config['num_classes']
    print("numclass:", num_classes)
    #determine all of the eval_classes
    #always ignoring 0 (background)
    if eval_classes is None:
        if num_classes == 1:
            eval_classes = [1]
        else:
            eval_classes = list(range(1, num_classes))
    eval_tfs = Compose([
        Normalize(mean=norms['mean'], std=norms['std']),
        ToTensorV2()
    ])

    #create the dataset and dataloader
    test_data = SegmentationData(test_dir, tfs=eval_tfs)
    test = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    inference_only = False

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
    
    #set the model to eval mode
    model.eval()
    image_ious = []
    num = 0 
    for data in test:
        num = num + 1
        image = data['image'].cuda(non_blocking=True) #(1, 1, H', W')
        mask = data['mask'].cuda(non_blocking=True)
        with torch.no_grad():
            prediction = sliding_window_inference(image, (224, 224), 4, model, overlap=0.7).detach()
            if num_classes == 1:
                #(1, 1, H, W) --> (H, W)
                prediction = (nn.Sigmoid()(prediction) > threshold).squeeze().cpu().numpy()
                prediction_img = (prediction * 255).astype(np.uint8)
                image = Image.fromarray(prediction_img)
                image.save(os.path.join(save_dir, data['fname'][0]))
            else:
                prediction = nn.Softmax(dim=1)(prediction) #(1, C, H, W)
                prediction = torch.argmax(prediction, dim=1) #(1, H, W)
                prediction = prediction.squeeze().cpu().numpy() #(H, W)

        if not inference_only:
            mask = data['mask'].squeeze().numpy()  #(1, H, W) --> (H, W)
            
            class_ious = []
            for label in eval_classes:
                #only consider areas in the mask and prediction
                #corresponding to the current label
                label_mask = mask == label
                label_pred = prediction == label
                
                #there's 1 hiccup to consider that occurs
                #when not all of the instances of an object
                #are labeled in a 2d image. this is the case for all
                #the datasets in the Perez benchmark (e.g. there may be
                #20 mitochondria in an image, but only 10 were labeled).
                #to handle this case we'll need to only consider parts of the
                #prediction that have some overlap with the ground truth. in
                #principle this will hide some instances that are FP, but it will
                #not hide FP pixels that were predicted as part of an instance
                if instance_match:
                    #because we're working with labeled instances
                    #we will label each connected component in the mask
                    #as a separate object
                    instance_label_mask = measure.label(label_mask)
                    instance_label_pred = measure.label(label_pred)

                    #we're going to evaluate IoU over all the pixels for
                    #the current label within the image
                    instance_matched_prediction = np.zeros_like(label_pred)
                    mask_instance_labels = np.unique(instance_label_mask)[1:]
                    for mask_instance_label in mask_instance_labels:
                        #find all the instance labels in the prediction that 
                        #coincide with the current mask instance label
                        prediction_instance_labels = np.unique(
                            instance_label_pred[instance_label_mask == mask_instance_label]
                        )

                        #add all pixels in the prediction with the detected instance
                        #labels to the instance_matched_prediction
                        for prediction_instance_label in prediction_instance_labels:
                            if prediction_instance_label != 0: #ignore background
                                instance_matched_prediction += instance_label_pred == prediction_instance_label

                    #alright, now finally, we can compare the label_mask and the instance_matched_prediction
                    intersect = np.logical_and(instance_matched_prediction, label_mask).sum()
                    union = np.logical_or(instance_matched_prediction, label_mask).sum()
                
                else:
                    #this is the case that we hope to find ourselves in.
                    #evaluation is much easier here
                    intersect = np.logical_and(label_pred, label_mask).sum()
                    union = np.logical_or(label_pred, label_mask).sum()
                
                class_ious.append((intersect + 1e-5) / (union + 1e-5)) # iou for each class (label) of the current image

            image_ious.append(class_ious) # every value in image_ious is a list with iou of each class


    
    #report the mean IoU
    if not inference_only:
        image_ious = np.array(image_ious)
        mean_class_ious = image_ious.mean(axis=0)
        for label, mci in zip(eval_classes, mean_class_ious):
            print(f'Class {label} IoU 2d: {mci}')
            
        #print the overall mean
        mean_iou = image_ious.mean()
        print(f'Mean IoU: {mean_iou}')
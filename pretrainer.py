"""
The main code to pre-train RETINA on CEM500K dataset. 
Including augmentations and training each iteration. 
"""
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from albumentations.pytorch import ToTensorV2
import torch.backends.cudnn as cudnn
from albumentations import ( 
    Compose, PadIfNeeded, Normalize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast,
    CropNonEmptyMaskIfExists, GaussNoise, RandomResizedCrop, Rotate, GaussianBlur, CoarseDropout
)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import cv2
import torchvision.transforms as tf
from PIL import ImageFilter, Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class SegmentationData(Dataset):
    def __init__(self, data_dir, tfs1=None, tfs2=None, tfs3=None):
        super(SegmentationData, self).__init__()
        self.data_dir = data_dir
        self.impath = data_dir
        self.fnames = next(os.walk(self.impath))[2]
        self.fnames = [fn for fn in self.fnames if fn[0] != '.']
        print(f'Found {len(self.fnames)} images in {self.impath}')
        
        self.tfs1 = tfs1
        self.tfs2 = tfs2
        self.tfs3 = tfs3
        self.gray_channels = 1
        
    def __len__(self):  
        return len(self.fnames)
    
    def __getitem__(self, idx):
        # we want the input is the augmented image, and the mask is the original image
        # Task is to reconstruct (recover) the augmented image to original image
        f = self.fnames[idx]
        image = cv2.imread(os.path.join(self.impath, f), 0)
        output1 = {'image': image}
        # apply randomcrop (tfs1) to both image and mask, 
        # but only apply the rest (tfs2) to inputimage, not for mask
        if self.tfs1 is not None:
            transformed1 = self.tfs1(**output1)
            output1['image'] = transformed1['image']
            mask = output1['image']
            output2 = {'image': output1['image']}
        else:
            print("the tfs1 is none!")
        if self.tfs2 is not None:
            transformed2 = self.tfs2(**output2)
            output2['image'] = transformed2['image']
        else:
            print("the tfs2 is none!")
        output_image = {'image': output2['image']}
        output_mask = {'image': mask}
        if self.tfs3 is not None:
            transformed_image = self.tfs3(**output_image)
            transformed_mask = self.tfs3(**output_mask)
            output_image['image'] = transformed_image['image']
            output_mask['image'] = transformed_mask['image']
        else:
            print("the tfs3 is none!")
        return {'image': output_image['image'], 'mask':output_mask['image']}
    
class DataFetcher:
    """
    Loads batches of images and masks from a dataloader onto the gpu.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.loader_iter = iter(dataloader)

    def __len__(self):
        return len(self.dataloader)
    
    def reset_loader(self):
        self.loader_iter = iter(self.dataloader)

    def load(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.reset_loader()
            batch = next(self.loader_iter)
            
        #get the images and masks as cuda float tensors
        images = batch['image'].float().cuda(non_blocking=True)

        masks = batch['mask'].float().cuda(non_blocking=True)
        #print("images:", images.shape)
        return images, masks

def trainer_synapse(args, model, rank):
    # rank is the local rank
    from datasets.dataset_synapse import RandomGenerator
    base_lr = args.base_lr
    # the mean and std come from the mocov2 config in cem500k paper. 
    normalize = Normalize(mean=0.57287007, std=0.12740536)
    print("the augmentaion has flip rotation and coarsedropout")
    augs1 = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=90, p=0.5),
        RandomResizedCrop(height=224, width=224, scale=(0.08,1.0), ratio=(0.5, 1.5)),
    ])
    augs2 = Compose([
        RandomBrightnessContrast(brightness_limit=0.3,contrast_limit=0.3),
        GaussNoise(var_limit=args.noise_range),
        GaussianBlur(),
        # randomly masks out rectangular regions of the input image 
        # by setting pixels within these regions to a specified value
        CoarseDropout(max_holes=args.max_holes, max_height=16, max_width=16, min_holes=1, fill_value=0, always_apply=False),
    ])
    augs3 = Compose([
        normalize,
        ToTensorV2()
    ])
    device = torch.device(f'cuda:{rank}')
    db_train = SegmentationData(data_dir=args.root_path, tfs1=augs1, tfs2=augs2, tfs3=augs3)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
    # local batch size is the global batch size divided by GPU number.
    local_batch_size = args.batch_size // args.world_size
    # number workers should be cpu number
    train = DataLoader(db_train, batch_size=local_batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=24, sampler=train_sampler)
    print("batchsize:", args.batch_size)
    print("The length of train set is: {}".format(len(train)))
    trainloader = DataFetcher(train)
    mse_loss = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    inner_loop = range(len(trainloader))
    model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model.train()
    
    for epoch_num in iterator:
        lossall = list()
        for iteration in inner_loop:
            images, masks = trainloader.load()        
            images.to(device)
            masks.to(device)
            outputs = model(images)
            loss = mse_loss(outputs[:].float(), masks[:].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossall.append(loss.item())
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1
        epochloss = sum(lossall) / len(lossall)
        print("loss_ave:", epochloss)
        print("learningrate:", lr_)
        print("epoch:", epoch_num)
        pretrained_num = 0
        if epoch_num % 20 == 0 and epoch_num != 0: 
            state_dict = model.module.state_dict()
            filename = f"/checkpoint/pretrain_{epoch_num + pretrained_num}.pt"
            torch.save(state_dict, filename)
            print("Saving checkpoint:", filename)
            x_detached = images.detach()  # Detach the tensor from the computation graph
            x_cpu = x_detached.cpu()  # Move the tensor to CPU
            x_numpy = x_cpu.numpy()
            inputname = f"/checkpoint/pics/input_{epoch_num+pretrained_num}.png"
            plt.imshow(x_numpy[0][0], cmap='gray')
            plt.axis('off')
            plt.savefig(inputname)
            plt.close()
            x_detached = outputs.detach()  # Detach the tensor from the computation graph
            x_cpu = x_detached.cpu()  # Move the tensor to CPU
            x_numpy = x_cpu.numpy()
            outputname = f"/checkpoint/pics/output_{epoch_num+pretrained_num}.png"
            plt.imshow(x_numpy[0][0], cmap='gray')
            plt.axis('off')
            plt.savefig(outputname)
            plt.close()
            x_detached = masks.detach()  # Detach the tensor from the computation graph
            x_cpu = x_detached.cpu()  # Move the tensor to CPU
            x_numpy = x_cpu.numpy()
            maskname = f"/checkpoint/pics/mask_{epoch_num+pretrained_num}.png"
            plt.imshow(x_numpy[0][0], cmap='gray')
            plt.axis('off')
            plt.savefig(maskname)
            plt.close()
    return epochloss
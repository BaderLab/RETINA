"""
Contains the function of loading the dataset.
Including data augmentations.
"""

import cv2, os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from albumentations import DualTransform
from albumentations import Resize

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['mask']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'mask': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None): # split: 'train'
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
class SegmentationData(Dataset):
    def __init__(self, data_dir, segmentation_classes=1, tfs=None, train_length=1):
        super(SegmentationData, self).__init__()
        self.data_dir = data_dir
        self.impath = os.path.join(data_dir, 'images')
        self.mskpath = os.path.join(data_dir, 'masks')
        if not os.path.isdir(self.mskpath):
            self.has_masks = False
            print(f'{self.mskpath} does not exist, dataset will be inference only!')
        
        #get filenames and remove any hidden files 
        #that might be in the directory
        self.fnames = next(os.walk(self.impath))[2]
        self.fnames = [fn for fn in self.fnames if fn[0] != '.']
        print(f'Found {len(self.fnames)} images in {self.impath}')
        
        self.tfs = tfs
        self.gray_channels = 1
        self.segmentation_classes = segmentation_classes
        self.train_length = train_length
            
    def __len__(self):
        return int(len(self.fnames) * self.train_length)
    
    def __getitem__(self, idx):
        #load the image and mask files
        f = self.fnames[idx]
        image = cv2.imread(os.path.join(self.impath, f), 0)
        mask = cv2.imread(os.path.join(self.mskpath, f), 0)
        output = {'image': image, 'mask': mask, 'fname':f}
        
        #apply transforms, assumes albumentations
        if self.tfs is not None:
            transformed = self.tfs(**output)
            output['image'] = transformed['image']
            output['mask'] = transformed['mask']
        
        if self.segmentation_classes > 1:
            output['mask'] = output['mask'].long()
        else:
            output['mask'] = output['mask'].long()
        return output
    
class FactorResize(DualTransform):
    """
    Resizes an image, but not the mask, to be divisible by a specific
    number like 32. Necessary for evaluation with segmentation models
    that use downsampling.
    """

    def __init__(self, resize_factor, always_apply=False, p=1.0):
        super(FactorResize, self).__init__(always_apply, p)
        self.rf = resize_factor

    def apply(self, image, **params):
        h, w = image.shape[:2]
        nh = int(h / self.rf) * self.rf
        nw = int(w / self.rf) * self.rf
        interpolation = cv2.INTER_LINEAR
        resize_transform = Resize(nh, nw, interpolation=interpolation)
        transformed = resize_transform(image=image)
        return transformed['image']

    def apply_to_mask(self, mask, **params):
        h, w = mask.shape[:2]
        nh = int(h / self.rf) * self.rf
        nw = int(w / self.rf) * self.rf
        interpolation = cv2.INTER_NEAREST
        resize_transform = Resize(nh, nw, interpolation=interpolation)
        transformed = resize_transform(image=mask)
        return transformed['image']
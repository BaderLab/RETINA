"""
This is the trainer file for fine-tuning run by train.py.
It contains dataset extraction and training for each iteration. 
We took the reference on TransUNet model:
(Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., et al. (2021).)
"""
import random, os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from utils import DiceLoss, BinaryDiceLoss
from torchvision import transforms
from albumentations import ( 
    Compose, PadIfNeeded, Normalize, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, 
     GaussNoise, RandomResizedCrop, Rotate, GaussianBlur, CropNonEmptyMaskIfExists
)
from albumentations.pytorch import ToTensorV2
from metrics import ComposeMetrics, IoU, EMAMeter, AverageMeter
from monai.inferers import sliding_window_inference
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP

augmentation_dict = {
    'PadIfNeeded': PadIfNeeded, 'HorizontalFlip': HorizontalFlip, 'VerticalFlip': VerticalFlip,
    'RandomBrightnessContrast': RandomBrightnessContrast, 'CropNonEmptyMaskIfExists': CropNonEmptyMaskIfExists,
    'GaussNoise': GaussNoise, 'RandomResizedCrop': RandomResizedCrop, 'Rotate': Rotate, 
    'GaussianBlur': GaussianBlur
}

def trainer_synapse(args, model, rank, train_length = 1):
    from datasets.dataset_synapse import SegmentationData, FactorResize, RandomGenerator
    base_lr = args.base_lr
    num_classes = args.num_classes
    dataset_augs = []
    for aug_params in args.augmentations:
        aug_name = aug_params['aug']
        
        #lookup aug_name and replace it with the 
        #correct augmentation class
        aug = augmentation_dict[aug_name]
        
        #delete the aug key and then the remaining
        #dictionary items are kwargs
        del aug_params['aug']
        dataset_augs.append(aug(**aug_params))
        
    #unpack the list of dataset specific augmentations
    #into Compose, and then add normalization and tensor
    #conversion, which apply universally
    normalize = Normalize(mean=args.mean, std=args.std)
    augs = Compose([
        *dataset_augs,
        normalize,
        ToTensorV2()
    ])
    
    eval_augs = Compose([
        FactorResize(32),
        normalize,
        ToTensorV2()
    ])
    device = torch.device(f'cuda:{rank}')
    # max_iterations = args.max_iterations
    db_train = SegmentationData(data_dir=args.root_path,segmentation_classes=num_classes, tfs=augs, train_length=train_length)
    print("The length of train set is: {}".format(len(db_train)))
    if args.valid_path != None:
        db_valid = SegmentationData(data_dir=args.valid_path, segmentation_classes=num_classes,tfs=eval_augs)
        print("The length of valid set is: {}".format(len(db_valid)))
    train_sampler = torch.utils.data.distributed.DistributedSampler(db_train)
    # local batch size is the global batch size divided by GPU number.
    local_batch_size = args.batch_size // args.ngpus
    print("local_batch_size:", local_batch_size)
    trainloader = DataLoader(db_train, batch_size=local_batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=16, sampler=train_sampler)
    if args.valid_path != None:
        validloader = DataLoader(db_valid, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    if num_classes > 1:       
        if args.class_weights != None:
            weight = torch.Tensor(args.class_weights).float().cuda()  
        else:
            weight = None
        lossCriterion = CrossEntropyLoss(weight=weight).cuda()
        dice_loss = DiceLoss(num_classes).cuda()
    else:
        lossCriterion = BCEWithLogitsLoss().cuda()
        dice_loss = BinaryDiceLoss().cuda()

    print(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    max_epoch = args.max_iterations // len(trainloader)
    iterator = tqdm(range(args.finetuned // len(trainloader), max_epoch + 1), ncols=70)
    trn_md = {'iou': IoU(EMAMeter())}
    class_names = args.class_names
    trn_metrics = ComposeMetrics(trn_md, class_names)
    trn_loss_meter = EMAMeter()
    val_md = {'iou': IoU(EMAMeter())}
    val_metrics = ComposeMetrics(val_md, class_names)
    val_loss_meter = AverageMeter()
    model.to(device)
    #rank should be the local rank
    model = DDP(model, device_ids=[rank], output_device=rank)
    model.train()
    iter_num = args.finetuned
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            model.train()
            outputs = model(image_batch)
            #create the loss criterion
            if num_classes > 1:
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss_criterion = lossCriterion(outputs, label_batch[:].long())
            else:
                loss_dice = dice_loss(outputs, label_batch)
                loss_criterion = lossCriterion(outputs, label_batch.unsqueeze(1).float())
                
            loss = 0.5 * loss_criterion + 0.5 * loss_dice
            
            optimizer.zero_grad()
            trn_loss_meter.update(loss.item())
            trn_metrics.evaluate(outputs.detach(), label_batch)
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
        if epoch_num % (max_epoch / args.num_prints) == 0 and epoch_num != 0:
            print('\n')
            print('iteration:', iter_num)
            print(f'train_loss: {trn_loss_meter.avg:.3f}')
            trn_loss_meter.reset()
            #prints and automatically resets the metric averages to 0
            trn_metrics.print()
            if args.valid_path != None:
                cudnn.benchmark = False
                model.eval()
            
                for i_batch, sampled_batch in enumerate(validloader):
                    with torch.no_grad():
                        image_batch, label_batch = sampled_batch['image'], sampled_batch['mask']
                        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                        #print("imageshape:", image_batch.shape)
                        output = sliding_window_inference(image_batch, (224, 224), 4, model, overlap=0.2)
                        if num_classes > 1:
                            loss = lossCriterion(output, label_batch[:].long())
                        else:
                            loss = lossCriterion(output, label_batch.unsqueeze(1).float())
                        val_loss_meter.update(loss.item())
                        val_metrics.evaluate(output.detach(), label_batch)
                print('\n') 
                print(f'valid_loss: {val_loss_meter.avg:.3f}')
                val_loss_meter.reset()
                val_metrics.print()       
                #turn cudnn.benchmark back on before returning to training
                cudnn.benchmark = True

        '''
        if iter_num % args.save_interval == 0 and epoch_num != 0:
            save_mode_path = os.path.join(args.model_dir, 'iteration_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
        '''
    return "Training Finished!"

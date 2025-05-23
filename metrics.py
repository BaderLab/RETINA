"""
This file contains the metrics of model accuracy measurement. 
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class AverageMeter:
    """Computes average values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = self.sum + self.val
        self.count += 1
        self.avg = self.sum / self.count

class EMAMeter:
    """Computes and stores an exponential moving average and current value"""
    def __init__(self, momentum=0.98):
        self.mom = momentum
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum = (self.sum * self.mom) + (val * (1 - self.mom))
        self.count += 1
        self.avg = self.sum / (1 - self.mom ** (self.count))

class IoU:
    """
    Give a meter object, calculates and stores average IoU results
    from batches of train or evaluation data.
    
    """
    
    def __init__(self, meter):
        self.meter = meter
        
    def calculate(self, output, target):
        #make target the same shape as output by unsqueezing
        #the channel dimension, if needed, 
        if target.ndim == output.ndim - 1:
            target = target.unsqueeze(1)
        # now the dimension is like [16, 7, H, W]
        
        #get the number of classes from the output channels
        n_classes = output.size(1)
        #get reshape size based on number of dimensions
        #can exclude first 2 dims, which are always batch and channel
        empty_dims = (1,) * (target.ndim - 2)
        
        if n_classes > 1:
            #softmax the output and argmax,  
            # The softmax function converts these scores into probabilities, 
            # making the sum of probabilities for each class at each pixel equal to 1.
            output = nn.Softmax(dim=1)(output) #(B, NC, H, W)
            #find the index (class) with the maximum probability for each pixel
            #dimension (NC) is reduced to 1, as argmax returns the index of the maximum class probability for each pixel
            max_idx = torch.argmax(output, 1, keepdim=True) #(B, 1, H, W)
            
            #one hot encode the target and output
            k = torch.arange(0, n_classes).view(1, n_classes, *empty_dims).to(target.device)
            target = (target == k)
            output_onehot = torch.zeros_like(output)
            # should be (B, NC, H, W)
            output_onehot.scatter_(1, max_idx, 1)
        else:
            #just sigmoid the output
            output = (nn.Sigmoid()(output) > 0.5).long()
        
        x_detached = output.detach()  # Detach the tensor from the computation graph
        x_numpy = x_detached.cpu().numpy()
            # Plotting
        plt.imshow(x_numpy[0][0], cmap='gray')
        plt.title("Sample Torch Image")
        plt.savefig("/kasthuri_pp/transunetoutput.png")
        plt.close()
        #cast target to the correct type for operations
        target = target.type(output.dtype)
        
        #multiply the tensors, everything that is still as 1 is 
        #part of the intersection (N,)
        # dims should be (0,2,3)
        dims = (0,) + tuple(range(2, target.ndimension()))
        # total intersection across all images in the batch and across all pixels, but separately for each class
        intersect = torch.sum(output * target, dims)
        
        #compute the union, (N,)
        union = torch.sum(output + target, dims) - intersect
        
        #avoid division errors by adding a small epsilon
        #if intersect and union are zero, then iou is 1
        iou = (intersect + 1e-5) / (union + 1e-5)
        
        return iou
    
    def update(self, value):
        self.meter.update(value)
        
    def reset(self):
        self.meter.reset()
        
    def average(self):
        return self.meter.avg
        
class ComposeMetrics:
    """
    A class for composing metrics together.
    
    Arguments:
    ----------
    
    metrics_dict: A dictionary in which each key is the name of a metric and
    each value is a metric object.
    
    class_names: Optional list of names for each class index (e.g. mitochondria, lysosomes, etc.)
    
    reset_on_print: Boolean. Whether to reset the results for each metric after the print
    function is called. Default True.
    
    """
    def __init__(self, metrics_dict, class_names=None, reset_on_print=True):
        self.metrics_dict = metrics_dict
        self.class_names = class_names
        self.reset_on_print = reset_on_print
        
    def evaluate(self, output, target):
        #calculate all the metrics in the dict
        # calculate IoU, each metric is iou(EMAmeter()) instance
        for metric in self.metrics_dict.values():
            value = metric.calculate(output, target)
            metric.update(value)
            
    def print(self):
        names = []
        values = []
        print("items:", self.metrics_dict.items())
        for name, metric in self.metrics_dict.items(): # is IoU and IoU metric object inside
            avg_values = metric.average()
            if isinstance(avg_values, torch.Tensor):
                avg_values = avg_values.cpu()
            # Now, avg_values is either an int or a CPU tensor
            #we expect metric to be a tensor of size (n_classes,)
            #we want to print the corresponding class names if given
            if self.class_names is None:
                self.class_names = [f'class_{i}' for i in range(len(avg_values))]
            
            #print("class_name", self.class_names)
            #print("avg", avg_values)
            for class_name, val in zip(self.class_names, avg_values):
                #print("name&value:", class_name)
                #print(val)
                names.append(f'{class_name}_{name}') # name is IoU
                values.append(val.item())
                
            if self.reset_on_print:
                metric.reset()
        
        for name, value in zip(names, values):
            print(f'{name}: {value:.3f}')
        
        mean_iou = np.array(values).mean()
        print(f'Mean IoU: {mean_iou}')
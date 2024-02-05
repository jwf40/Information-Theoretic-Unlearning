"""
This file is used for the Selective Synaptic Dampening method
Strategy files use the methods from here
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, dataset, TensorDataset
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import os
import pdb
import math
import shutil
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import scipy.stats as stats
from typing import Dict, List
from torchvision.transforms import v2
from models import ViT
###############################################
# Clean implementation
###############################################

#https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device='cpu'):
        self.std = std
        self.mean = mean
        self.device = device
        
    def __call__(self, tensor):
        _max = tensor.max()
        _min = tensor.min()
        tensor = tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
        tensor = torch.clamp(tensor, min=_min, max=_max)
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddUniformNoise(object):
    def __init__(self, min=-0.5, max=0.5, device='cpu'):
        self.min = min
        self.max = max
        self.device = device
        
    def __call__(self, tensor):
        return tensor + ((self.min - self.max) * torch.rand(tensor.size()).to(self.device) + self.max)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddSnPNoise(object):
    def __init__(self, perc=0.1, device='cpu'):
        self.perc = perc        
        self.device = device
        
    def __call__(self, tensor):
        perm = torch.randperm(tensor.size(0))
        k = len(perm)*self.perc
        idx = perm[:k]
        tensor[:(idx//2)] = tensor.min()
        tensor[(idx//2):] = tensor.max()
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class Lipschitz:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        #self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        #self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]
        self.n_epochs = parameters["n_epochs"]
        self.n_samples = parameters["n_samples"]
        self.learning_rate = parameters["learning_rate"]
        self.use_quad_weight = parameters["use_quad_weight"]
        self.ewc_lambda = parameters["ewc_lambda"]
        self.lipschitz_weighting = parameters["lipschitz_weighting"]
        print(type(self.model))
        crop_size = (224, 224) if isinstance(self.model, ViT) else (32,32)
        # self.transforms = v2.Compose([
        #     v2.RandomResizedCrop(size=(crop_size), antialias=True),
        #     v2.RandomHorizontalFlip(p=0.5),
        #     v2.RandomAffine(45),
        #     v2.ColorJitter(brightness=10, contrast=10, saturation=10, hue=0.5),                        
        #     v2.ToTensor(),
        # ])

        self.transforms = v2.Compose([
            AddGaussianNoise(0., self.lipschitz_weighting, device=self.device), #0.1
            v2.ToTensor(),
        ])

        # self.transforms = v2.Compose([
        #     AddUniformNoise(min= -self.lipschitz_weighting, max=self.lipschitz_weighting, device=self.device), #0.1
        #     v2.ToTensor(),
        # ])

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def fulllike_params_dict(
        self, model: torch.nn, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: list) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def subsample_dataset(self, dataset: dataset, sample_perc: float) -> Subset:
        """
        Take a subset of the dataset

        Parameters:
        dataset (dataset): dataset to be subsampled
        sample_perc (float): percentage of dataset to sample. range(0,1)
        Returns:
        Subset (float): requested subset of the dataset
        """
        sample_idxs = np.arange(0, len(dataset), step=int((1 / sample_perc)))
        return Subset(dataset, sample_idxs)

    def split_dataset_by_class(self, dataset: dataset) -> List[Subset]:
        """
        Split dataset into list of subsets
            each idx corresponds to samples from that class

        Parameters:
        dataset (dataset): dataset to be split
        Returns:
        subsets (List[Subset]): list of subsets of the dataset,
            each containing only the samples belonging to that class
        """
        n_classes = len(set([target for _, target in dataset]))
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (x, y) in enumerate(dataset):
            subset_idxs[y].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)]

    def classwise_datasets(self, dataset):
        class_ids = np.unique(dataset.targets)
        class_datasets = []
        for class_id in class_ids:
            idx = np.where(dataset.targets == class_id)[0]
            class_datasets.append(Subset(dataset, idx))
        
        return class_datasets               

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            x, _, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            #loss = criterion(out, y)
            loss = torch.norm(out, p="fro", dim=1).pow(2).mean()
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    #imp.data += p.grad.data.clone().pow(2)
                    imp.data += p.grad.data.clone().abs()

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        forget_dl: DataLoader,                
    ) -> None:
        """
        Spectral forgetting but first perturb weights based on the SSD equations given in the paper
        Parameters:        
               

        Returns:
        None
        """
       

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)           
        #Calculate smoothness penalty
        self.opt.zero_grad()

        for x,_, y in forget_dl:            
            x = x.to(self.device)
            image = x.unsqueeze(0) if x.dim() == 3 else x
            out = self.model(image)                            
            loss = torch.tensor(0.0, device=self.device)
            out_n = torch.tensor(0.0, device=self.device)
            in_n = torch.tensor(0.0, device=self.device)
            #Build comparison images
            
            for _ in range(self.n_samples):   
                img2 = self.transforms(deepcopy(x))             
                image2 = img2.unsqueeze(0) if img2.dim() == 3 else img2
                
                with torch.no_grad():
                    out2 = self.model(image2)
                # out2 = self.model(image2)
                #ignore batch dimension        
                flatimg, flatimg2 = image.view(image.size()[0], -1), image2.view(image2.size()[0], -1)

                in_norm = torch.linalg.vector_norm(flatimg - flatimg2, dim=1)              
                out_norm = torch.linalg.vector_norm(out - out2, dim=1)
                #these are used to print
                in_n += in_norm.sum()
                out_n += out_norm.sum()
                #K = 0.001 * ((0.4- (out_norm / in_norm)).sum()).abs()#1*((0.08-
                K =  ((out_norm / in_norm).sum()).abs()#pow(2)#  0.1                                                
                loss += K
            
            #Normalize
            loss /= self.n_samples
            in_n /= self.n_samples 
            out_n /= self.n_samples
            print(loss)            
            print(in_n)
            print(out_n)
            # sys.exit()
            loss.backward()      
            optimizer.step()

        
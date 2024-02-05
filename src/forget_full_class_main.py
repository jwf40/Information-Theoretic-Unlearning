#!/bin/python3.8

import random
import os
import wandb
#import optuna
from typing import Tuple, List
import sys
import argparse
import time
from datetime import datetime
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import models
from unlearn import *
from utils import *
import forget_full_class_strategies
import datasets
import models
import conf
from training_utils import *
import optuna

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-weight_path', type=str, required=True, help='Path to model weights. If you need to train a new model use pretrain_model.py')
parser.add_argument('-dataset', type=str, required=True, nargs='?',
                    choices=['Cifar10', 'Cifar20', 'Cifar100', 'PinsFaceRecognition'],
                    help='dataset to train on')
parser.add_argument('-classes', type=int, required=True,help='number of classes')
parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('-method', type=str, required=True, nargs='?',
                    choices=['baseline', 'retrain','finetune','blindspot','amnesiac','UNSIR', 'ssd_tuning', 'graceful_forgetting', 'lipschitz_forgetting', 'scrub', 'gkt', 'emmn'],
                    help='select unlearning method from choice set')    
parser.add_argument('-forget_class', type=str, required=True,nargs='?',help='class to forget',
                    choices=list(conf.class_dict))
parser.add_argument('-epochs', type=int, default=1, help='number of epochs of unlearning method to use')
parser.add_argument("-seed", type=int, default=0, help="seed for runs")

args = parser.parse_args()

if __name__=='__main__':

    # # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    # Check that the correct things were loaded
    if args.dataset == "Cifar20":
        assert args.forget_class in conf.cifar20_classes
    elif args.dataset == "Cifar100":
        assert args.forget_class in conf.cifar100_classes

    forget_class = conf.class_dict[args.forget_class]

    batch_size = args.b

    # Change alpha here as described in the paper
    # For PinsFaceRe-cognition, we use α=50 and λ=0.1
    model_size_scaler = 1
    if args.net == "ViT":
        model_size_scaler = 0.5
    else:
        model_size_scaler = 1


                
    # get network
    return_activations = (args.method=='gkt')
    net = getattr(models, args.net)(num_classes=args.classes, return_activations=return_activations)
    net.load_state_dict(torch.load(args.weight_path))
    

    # for bad teacher
    unlearning_teacher = getattr(models, args.net)(num_classes=args.classes)

    if args.gpu:
        net = net.cuda()
        unlearning_teacher = unlearning_teacher.cuda()

    # For celebritiy faces
    root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

    # Scale for ViT (faster training, better performance)
    img_size = 224 if args.net == "ViT" else 32
    trainset = getattr(datasets, args.dataset)(
        root=root, download=True, train=True, unlearning=True, img_size=img_size
    )
    validset = getattr(datasets, args.dataset)(
        root=root, download=True, train=False, unlearning=True, img_size=img_size
    )

    # Set up the dataloaders and prepare the datasets
    trainloader = DataLoader(trainset, batch_size=args.b, shuffle=True)
    
    full_train_dl = DataLoader(deepcopy(trainset), batch_size=args.b, shuffle=True)
    
    _sample = next(iter(trainloader))[0]
    print(_sample.min(), _sample.max(), _sample.mean())

    validloader = DataLoader(validset, batch_size=args.b, shuffle=False)

    classwise_train, classwise_test = forget_full_class_strategies.get_classwise_ds(
        trainset, args.classes
    ), forget_full_class_strategies.get_classwise_ds(validset, args.classes)

    (
        retain_train,
        retain_valid,
        forget_train,
        forget_valid,
    ) = forget_full_class_strategies.build_retain_forget_sets(
        classwise_train, classwise_test, args.classes, forget_class
    )
    forget_valid_dl = DataLoader(forget_valid, batch_size)
    retain_valid_dl = DataLoader(retain_valid, batch_size)

    forget_train_dl = DataLoader(forget_train, batch_size)
    retain_train_dl = DataLoader(retain_train, batch_size, shuffle=True)
    # full_train_dl = DataLoader(
    #     ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
    #     batch_size=batch_size,
    # )
    print(len(forget_train_dl))

    if args.net == 'ViT':
        damp_val = 1
        select_val = 3.5
        lipschitz_weighting = 0.8
        learning_rate = 1.5
        #Placeholder
        scrub_alpha= 10 #1
        scrub_gamma= 10 #5
    elif args.net == 'VGG16':
        damp_val = 4
        select_val = 10
        lipschitz_weighting = 0.5
        learning_rate = 0.0003
        scrub_alpha= 1
        scrub_gamma= 5

    parameters = {
        "dampening_constant": 1, 
        "selection_weighting": 1,
        'eps': 0.01,
        'use_quad_weight': False,    
        'n_epochs': 1,
        "ewc_lambda": 1, 
        "frequency": [0.038,0.0475, 0.1125],
        "amplitude": 0.1,
        'lipschitz_weighting': lipschitz_weighting,
        "learning_rate": learning_rate,
        "n_samples": 25
    }

    kwargs = {
        'model': net,
        'unlearning_teacher':unlearning_teacher, 
        'retain_train_dl': retain_train_dl,
        'retain_valid_dl': retain_valid_dl,
        'forget_train_dl': forget_train_dl,
        'forget_valid_dl': forget_valid_dl,
        'valid_dl': validloader,
        'dampening_constant': parameters["dampening_constant"],
        'selection_weighting': parameters["selection_weighting"],
        'eps': parameters["eps"],
        'use_quad_weight': parameters['use_quad_weight'],
        'n_epochs': parameters['n_epochs'],
        'forget_class': forget_class,
        'full_train_dl': full_train_dl,
        'num_classes': args.classes,
        'dataset_name': args.dataset,
        'device': 'cuda' if args.gpu else 'cpu',
        'model_name': args.net,
        "n_samples": parameters["n_samples"],
        'learning_rate': parameters["learning_rate"],
        "ewc_lambda": parameters["ewc_lambda"],
        "amplitude": parameters["amplitude"],
        "frequency": parameters["frequency"],
        "lipschitz_weighting": parameters['lipschitz_weighting']
    }
    #############
    

    ################

    wandb.init(project=f"PINSFINAL_LipschitzFinal_{args.net}_{args.dataset}_fullclass", name=f'{args.method}_{select_val}_{args.forget_class}')
    # Time the method
    import time

    start = time.time()

    testacc, retainacc, mia, d_f = getattr(forget_full_class_strategies, args.method)(**kwargs)

    end = time.time()
    time_elapsed = end - start
    wandb.log(
    {
        'TestAcc': testacc,
        'RetainTestAcc': retainacc,
        'MIA': mia,
        'df': d_f, 
        "MethodTime": time_elapsed,                   
    }
    )
    print("done logging...")
    wandb.finish()


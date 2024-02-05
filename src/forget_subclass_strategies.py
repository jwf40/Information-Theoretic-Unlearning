"""
Refer to forget_full_class_... for comments
This file is near identical with minimal modifications to facilitate subclass forgetting.
Seperate file to allow for easy reuse.
"""


import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset, TensorDataset

from sklearn import linear_model, model_selection
from tqdm import tqdm

from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import conf
import lipschitz

def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[label].append((img, label, clabel))
    return classwise_ds


def build_retain_forget_sets(
    classwise_train, classwise_test, num_classes, forget_class
):
    # Getting the forget and retain validation data
    forget_valid = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_test[cls]:
                forget_valid.append((img, label, clabel))

    retain_valid = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_test[cls]:
                retain_valid.append((img, label, clabel))

    forget_train = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_train[cls]:
                forget_train.append((img, label, clabel))

    retain_train = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, clabel))

    return (retain_train, retain_valid, forget_train, forget_valid)


def get_gf_loader(forget_train_dl):
    guided_x, guided_y = [], []
    for x,_, y in forget_train_dl.dataset:
        guided_x.append(x)
        guided_y.append(3)
    return DataLoader(TensorDataset(torch.stack(guided_x).squeeze(), torch.tensor(guided_y)), batch_size=forget_train_dl.batch_size)


def guided_eval(net, forget_train_dl, device, target_label=3):
    """
    What percentage of the forget set is now classified into any random class? *Unintended/uncontrolled change in behaviour
    """
    #gfdl = get_gf_loader(forget_train_dl)

    correct = 0.0
    total = 0.0
    for (x,_, y) in forget_train_dl:        
        x, y = x.to(device), y.to(device)
        preds = net(x)
        
        tmp, preds = preds.max(dim=1)
        total += y.size(dim=0)
        #where the predictions are safe
        t = torch.Tensor([target_label for _ in range(y.size(0))])
        t = t.to(device)
        correct += preds.eq(y).sum()  
        remaining_samples = torch.where(y!=t)
        correct += preds[remaining_samples].eq(t[remaining_samples]).sum()           
    #return the inverse
    correct /= total    
    incorrect = 1-correct
    return incorrect

def get_metric_scores(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
):
    loss_acc_dict = evaluate(model, valid_dl, device)
    retain_acc_dict = evaluate(model, retain_valid_dl, device)
    #zrf = UnLearningScore(model, unlearning_teacher, forget_valid_dl, 128, device)
    d_f = evaluate(model, forget_valid_dl, device)
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)
    # guided_acc = guided_eval(model, forget_train_dl, device)
    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], mia, d_f["Acc"])
    


def baseline(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def retrain(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dataset_name,
    model_name,
    device,
    **kwargs,
):
    for layer in model.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    if model_name == "ViT":
        epochs = getattr(conf, f"{dataset_name}_{model_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_{model_name}_MILESTONES")
    else:
        epochs = getattr(conf, f"{dataset_name}_EPOCHS")
        milestones = getattr(conf, f"{dataset_name}_MILESTONES")
    _ = fit_one_cycle(
        epochs,
        model,
        retain_train_dl,
        retain_valid_dl,
        milestones=milestones,
        device=device,
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def finetune(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    _ = fit_one_cycle(
        5, model, retain_train_dl, retain_valid_dl, lr=0.02, device=device
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def blindspot(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    **kwargs,
):
    student_model = deepcopy(model)
    KL_temperature = 1
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    retain_train_subset = random.sample(
        retain_train_dl.dataset, int(0.3 * len(retain_train_dl.dataset))
    )

    if kwargs["model_name"] == "ViT":
        b_s = 128  # lowered batch size from 256 (original) to fit into memory
    else:
        b_s = 256

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=model,
        retain_data=retain_train_subset,
        forget_data=forget_train_dl.dataset,
        epochs=1,
        optimizer=optimizer,
        lr=0.0001,
        batch_size=b_s,
        device=device,
        KL_temperature=KL_temperature,
    )

    return get_metric_scores(
        student_model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_superclasses,
    forget_superclass,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_superclasses))
    unlearning_trainset = []

    unlearninglabels.remove(forget_superclass)

    for x, y, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, y, random.choice(unlearninglabels)))

    for x, y, clabel in retain_train_dl.dataset:
        unlearning_trainset.append((x, y, clabel))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, 128, pin_memory=True, shuffle=True
    )

    _ = fit_one_unlearning_cycle(
        3, model, unlearning_train_set_dl, retain_valid_dl, device=device, lr=0.0001
    )
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_superclasses,
    device,
    **kwargs,
):
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, _, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = loss_fn(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)

    def get_mean_var(p, is_base_dist=False, alpha=3e-6):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_superclasses:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.ndim == 1:
            # BatchNorm
            var *= 10
        #         var*=1
        return mu, var

    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(retain_train_dl.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha=alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )


def UNSIR(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_subclasses,
    forget_subclass,
    forget_superclass,
    device,
    **kwargs,
):
    classwise_train = get_classwise_ds(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)),
        num_subclasses,
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_subclasses):
        if i != forget_subclass:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = forget_superclass
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 250, noise_batch_size, device=device
    )
    noisy_loader = UNSIR_create_noisy_loader(
        noise, forget_class_label, retain_samples, noise_batch_size, device=device
    )
    # impair step
    _ = fit_one_unlearning_cycle(
        1, model, noisy_loader, retain_valid_dl, device=device, lr=0.0001
    )
    # repair step
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append(
            (
                retain_samples[i][0].cpu(),
                torch.tensor(retain_samples[i][2]),
                torch.tensor(retain_samples[i][2]),
            )
        )

    heal_loader = torch.utils.data.DataLoader(
        other_samples, batch_size=128, shuffle=True
    )
    _ = fit_one_unlearning_cycle(
        1, model, heal_loader, retain_valid_dl, device=device, lr=0.0001
    )

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

# SSD from https://github.com/if-loops/selective-synaptic-dampening/blob/main/src/ssd.py
def ssd_tuning(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = pdr.calc_importance(forget_train_dl)
    original_importances = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importances, sample_importances)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )

def lipschitz_forgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    dampening_constant,
    selection_weighting,
    device,
    use_quad_weight=False,
    n_epochs=5,
    learning_rate=0.01,    
    n_samples=25,
    ewc_lambda=1,
    lipschitz_weighting=0.1,
    **kwargs
):
    parameters = {
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Alpha from paper
        "n_epochs": n_epochs,
        "use_quad_weight": use_quad_weight,
        "ewc_lambda":ewc_lambda,
        "lipschitz_weighting":lipschitz_weighting,
        "learning_rate": learning_rate,
        "n_samples": n_samples
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    pdr = lipschitz.Lipschitz(model, optimizer, device, parameters)   

    pdr.modify_weight(forget_train_dl)

    return get_metric_scores(
        model,
        unlearning_teacher,
        retain_train_dl,
        retain_valid_dl,
        forget_train_dl,
        forget_valid_dl,
        valid_dl,
        device,
    )



#################################################
################ SCRUB METHODS ##################
#################################################


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class Namespace():
    """
    Helper class to emulate an argparse namespace object
    """    
    def __init__(self)->None:
        pass
    def __str__(self) -> str:
        for key, val in self.__dict__.items():
            print(f'{key}: {val}')

def sgda_adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    new_lr = opt.sgda_learning_rate
    if steps > 0:
        new_lr = opt.sgda_learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

def param_dist(model, model_s, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), model_s.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist

def train_distill(epoch, train_loader, module_list, swa_model, criterion_list, optimizer, opt, split, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()


    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, _, target = data

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()
                index = index.cuda()

        # ===================forward=====================
        #feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        logit_s = model_s(input)
        with torch.no_grad():
            #feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            #feat_t = [f.detach() for f in feat_t]
            logit_t = model_t(input)


        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        else:
            raise NotImplementedError(opt.distill)

        if split == "minimize":
            loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, opt.smoothing)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

    return

def scrub( model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    device,
    scrub_alpha=0.001,
    scrub_gamma=0.99,
    **kwargs):
    
    args = Namespace()
    args.optim = 'sgd'
    args.gamma = scrub_gamma
    args.alpha = scrub_alpha
    args.beta = 0
    args.smoothing = 0.0
    args.msteps = 2
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4
    args.distill = 'kd'

    """
    IMPORTANT:     
    ¬¬ sgda_decay_epochs will need to < sgda_epochs (obviously)
    ¬¬ 
    """
    args.sgda_epochs = 3
    args.sgda_learning_rate = 0.0005
    args.lr_decay_epochs = [5,8,9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9

    model_t = deepcopy(model)
    model_s = deepcopy(model)
    
    beta = 0.1
    def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
        1 - beta) * averaged_model_parameter + beta * model_parameter
    swa_model = torch.optim.swa_utils.AveragedModel(
        model_s, avg_fn=avg_fn)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            momentum=args.sgda_momentum,
                            weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam": 
        optimizer = torch.optim.Adam(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = torch.optim.RMSprop(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            momentum=args.sgda_momentum,
                            weight_decay=args.sgda_weight_decay)
        
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        model_s.cuda()
    
    for epoch in range(1, args.sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        # print("==> scrub unlearning ...")

        # acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
        # acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
        # acc_rs.append(100-acc_r.item())
        # acc_fs.append(100-acc_f.item())

        maximize_loss = 0
        if epoch <= args.msteps:
            train_distill(epoch, forget_train_dl, module_list, swa_model, criterion_list, optimizer, args, "maximize")
        train_distill(epoch, retain_train_dl, module_list, swa_model, criterion_list, optimizer, args, "minimize",)
        if epoch >= args.sstart:
            swa_model.update_parameters(model_s)

    return get_metric_scores(
            model_s,
            unlearning_teacher,
            retain_train_dl,
            retain_valid_dl,
            forget_train_dl,
            forget_valid_dl,
            valid_dl,
            device,
        )

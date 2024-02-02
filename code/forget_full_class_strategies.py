import random
import numpy as np
from typing import Tuple, List
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, ConcatDataset, dataset, TensorDataset, Subset, SubsetRandomSampler
import itertools

from tqdm import tqdm
from sklearn import linear_model, model_selection
from collections import OrderedDict


from unlearn import *
from metrics import UnLearningScore, get_membership_attack_prob
from utils import *
import ssd as ssd
import lipschitz
from models import *
import conf
import timeit
from gkt import *

# Create datasets of the classes
def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in tqdm(range(num_classes), desc='getting classwise data'):
        classwise_ds[i] = []

    for img, label, clabel in tqdm(ds, desc='splitting classes'):
        classwise_ds[clabel].append((img, label, clabel))

    return classwise_ds


# Creates datasets for method execution
def build_retain_forget_sets(
    classwise_train, classwise_test, num_classes, forget_class
):
    # Getting the forget and retain validation data
    forget_valid = []
    for cls in tqdm(range(num_classes), desc='getting forget_valid set'):
        if cls == forget_class:
            for img, label, clabel in classwise_test[cls]:
                forget_valid.append((img, label, clabel))

    retain_valid = []
    for cls in tqdm(range(num_classes), desc='getting retain_valid set'):
        if cls != forget_class:
            for img, label, clabel in classwise_test[cls]:
                retain_valid.append((img, label, clabel))

    forget_train = []
    for cls in tqdm(range(num_classes), desc='getting forget_train set'):
        if cls == forget_class:
            for img, label, clabel in classwise_train[cls]:
                forget_train.append((img, label, clabel))

    retain_train = []
    for cls in tqdm(range(num_classes), desc='getting retain_train set'):
        if cls != forget_class:
            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, label, clabel))

    return (retain_train, retain_valid, forget_train, forget_valid)

# Returns metrics
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
    d_f = evaluate(model, forget_valid_dl, device)
    mia = get_membership_attack_prob(retain_train_dl, forget_train_dl, valid_dl, model)    
    return (loss_acc_dict["Acc"], retain_acc_dict["Acc"], mia, d_f["Acc"])


# Does nothing; original model
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


# Retrain the model on the retrain dataset only
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


# Finetune the model using the retain data for a set number of epochs
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


# Bad Teacher from https://github.com/vikram2000b/bad-teaching-unlearning
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


# Implementation from https://github.com/vikram2000b/bad-teaching-unlearning
def amnesiac(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    **kwargs,
):
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    unlearninglabels.remove(forget_class)

    for x, _, clabel in forget_train_dl.dataset:
        unlearning_trainset.append((x, _, random.choice(unlearninglabels)))

    for x, _, y in retain_train_dl.dataset:
        unlearning_trainset.append((x, _, y))

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


# Extremely slow >>> Fisher https://github.com/AdityaGolatkar/SelectiveForgetting
def NTK(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
    device,
    **kwargs,
):
    def delta_w_utils(model_init, dataloader, name="complete"):
        model_init.eval()
        dataloader = torch.utils.data.DataLoader(
            dataloader.dataset, batch_size=1, shuffle=False
        )
        G_list = []
        f0_minus_y = []
        for idx, batch in enumerate(
            tqdm(dataloader)
        ):  # (tqdm(dataloader,leave=False)):
            batch = [
                tensor.to(next(model_init.parameters()).device) for tensor in batch
            ]
            input, _, target = batch

            target = target.cpu().detach().numpy()
            output = model_init(input)
            G_sample = []
            for cls in range(num_classes):
                grads = torch.autograd.grad(
                    output[0, cls], model_init.parameters(), retain_graph=True
                )
                grads = np.concatenate([g.view(-1).cpu().numpy() for g in grads])
                G_sample.append(grads)
                G_list.append(grads)
                p = (
                    torch.nn.functional.softmax(output, dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                    .transpose()
                )
                p[target] -= 1
                f0_y_update = deepcopy(p)
            f0_minus_y.append(f0_y_update)
        return np.stack(G_list).transpose(), np.vstack(f0_minus_y)

    #############################################################################################
    model_init = deepcopy(model)
    G_r, f0_minus_y_r = delta_w_utils(deepcopy(model), retain_train_dl, "complete")
    print("GOT GR")
    # np.save('NTK_data/G_r.npy',G_r)
    # np.save('NTK_data/f0_minus_y_r.npy',f0_minus_y_r)
    # del G_r, f0_minus_y_r

    G_f, f0_minus_y_f = delta_w_utils(deepcopy(model), forget_train_dl, "retain")
    print("GOT GF")
    # np.save('NTK_data/G_f.npy',G_f)
    # np.save('NTK_data/f0_minus_y_f.npy',f0_minus_y_f)
    # del G_f, f0_minus_y_f

    # G_r = np.load('NTK_data/G_r.npy')
    # G_f = np.load('NTK_data/G_f.npy')
    G = np.concatenate([G_r, G_f], axis=1)
    print("GOT G")
    # np.save('NTK_data/G.npy',G)
    # del G, G_f, G_r

    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    # f0_minus_y_f = np.load('NTK_data/f0_minus_y_f.npy')
    f0_minus_y = np.concatenate([f0_minus_y_r, f0_minus_y_f])

    # np.save('NTK_data/f0_minus_y.npy',f0_minus_y)
    # del f0_minus_y, f0_minus_y_r, f0_minus_y_f

    weight_decay = 0.1

    # G = np.load('NTK_data/G.npy')
    theta = G.transpose().dot(G) + (
        len(retain_train_dl.dataset) + len(forget_train_dl.dataset)
    ) * weight_decay * np.eye(G.shape[1])
    # del G

    theta_inv = np.linalg.inv(theta)

    # np.save('NTK_data/theta.npy',theta)
    # del theta

    # G = np.load('NTK_data/G.npy')
    # f0_minus_y = np.load('NTK_data/f0_minus_y.npy')
    w_complete = -G.dot(theta_inv.dot(f0_minus_y))

    # np.save('NTK_data/theta_inv.npy',theta_inv)
    # np.save('NTK_data/w_complete.npy',w_complete)
    # del G, f0_minus_y, theta_inv, w_complete

    # G_r = np.load('NTK_data/G_r.npy')
    num_to_retain = len(retain_train_dl.dataset)
    theta_r = G_r.transpose().dot(G_r) + num_to_retain * weight_decay * np.eye(
        G_r.shape[1]
    )
    # del G_r

    theta_r_inv = np.linalg.inv(theta_r)
    # np.save('NTK_data/theta_r.npy',theta_r)
    # del theta_r

    # G_r = np.load('NTK_data/G_r.npy')
    # f0_minus_y_r = np.load('NTK_data/f0_minus_y_r.npy')
    w_retain = -G_r.dot(theta_r_inv.dot(f0_minus_y_r))

    # np.save('NTK_data/theta_r_inv.npy',theta_r_inv)
    # np.save('NTK_data/w_retain.npy',w_retain)
    # del G_r, f0_minus_y_r, theta_r_inv, w_retain

    def get_delta_w_dict(delta_w, model):
        # Give normalized delta_w
        delta_w_dict = OrderedDict()
        params_visited = 0
        for k, p in model.named_parameters():
            num_params = np.prod(list(p.shape))
            update_params = delta_w[params_visited : params_visited + num_params]
            delta_w_dict[k] = torch.Tensor(update_params).view_as(p)
            params_visited += num_params
        return delta_w_dict

    #### Scrubbing Direction
    # w_complete = np.load('NTK_data/w_complete.npy')
    # w_retain = np.load('NTK_data/w_retain.npy')
    print("got prelims, calculating delta_w")
    delta_w = (w_retain - w_complete).squeeze()
    print("got delta_w")
    # delta_w_copy = deepcopy(delta_w)
    # delta_w_actual = vectorize_params(model0)-vectorize_params(model)

    # print(f'Actual Norm-: {np.linalg.norm(delta_w_actual)}')
    # print(f'Predtn Norm-: {np.linalg.norm(delta_w)}')
    # scale_ratio = np.linalg.norm(delta_w_actual)/np.linalg.norm(delta_w)
    # print('Actual Scale: {}'.format(scale_ratio))
    # log_dict['actual_scale_ratio']=scale_ratio
    def vectorize_params(model):
        param = []
        for p in model.parameters():
            param.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(param)

    m_pred_error = (
        vectorize_params(model) - vectorize_params(model_init) - w_retain.squeeze()
    )
    print(f"Delta w -------: {np.linalg.norm(delta_w)}")

    inner = np.inner(
        delta_w / np.linalg.norm(delta_w), m_pred_error / np.linalg.norm(m_pred_error)
    )
    print(f"Inner Product--: {inner}")

    if inner < 0:
        angle = np.arccos(inner) - np.pi / 2
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.sin(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")
    else:
        angle = np.arccos(inner)
        print(f"Angle----------:  {angle}")

        predicted_norm = np.linalg.norm(delta_w) + 2 * np.cos(angle) * np.linalg.norm(
            m_pred_error
        )
        print(f"Pred Act Norm--:  {predicted_norm}")

    predicted_scale = predicted_norm / np.linalg.norm(delta_w)
    predicted_scale
    print(f"Predicted Scale:  {predicted_scale}")
    # log_dict['predicted_scale_ratio']=predicted_scale

    # def NIP(v1,v2):
    #     nip = (np.inner(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)))
    #     print(nip)
    #     return nip
    # nip=NIP(delta_w_actual,delta_w)
    # log_dict['nip']=nip
    scale = predicted_scale
    direction = get_delta_w_dict(delta_w, model)

    for k, p in model.named_parameters():
        p.data += (direction[k] * scale).to(device)

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


# From https://github.com/AdityaGolatkar/SelectiveForgetting
def FisherForgetting(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    forget_class,
    num_classes,
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
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())
        if p.size(0) == num_classes:
            mu[forget_class] = 0
            var[forget_class] = 0.0001
        if p.size(0) == num_classes:
            # Last layer
            var *= 10
        elif p.ndim == 1:
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


# Implementation from https://github.com/vikram2000b/Fast-Machine-Unlearning
def UNSIR(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    **kwargs,
):
    classwise_train = get_classwise_ds(
        ConcatDataset((retain_train_dl.dataset, forget_train_dl.dataset)), num_classes
    )
    noise_batch_size = 32
    retain_valid_dl = DataLoader(retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class
    num_samples = 500
    retain_samples = []
    for i in range(num_classes):
        if i != forget_class:
            retain_samples += classwise_train[i][:num_samples]

    forget_class_label = forget_class
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 25, noise_batch_size, device=device
    )
    noisy_loader = UNSIR_create_noisy_loader(
        noise,
        forget_class_label,
        retain_samples,
        batch_size=noise_batch_size,
        device=device,
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
        "lower_bound": 1,  # unused
        "exponent": 1,  # unused
        "magnitude_diff": None,  # unused
        "min_layer": -1,  # -1: all layers are available for modification
        "max_layer": -1,  # -1: all layers are available for modification
        "forget_threshold": 1,  # unused
        "dampening_constant": dampening_constant,  # Lambda from paper
        "selection_weighting": selection_weighting,  # Alpha from paper
    }

    print(parameters['dampening_constant'], parameters["selection_weighting"])

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ssd.ParameterPerturber(model, optimizer, device, parameters)

    model = model.eval()

    # Calculation of the forget set importances
    sample_importances = pdr.calc_importance(forget_train_dl)

    # Calculate the importances of D (see paper); this can also be done at any point before forgetting.
    original_importances = pdr.calc_importance(full_train_dl)

    # Dampen selected parameters
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
    full_train_dl,
    device,
    num_classes,
    eps=1e-45,    
    use_quad_weight=False,
    n_epochs=5,
    n_samples=25,
    learning_rate=0.01,    
    ewc_lambda=1,
    lipschitz_weighting=0.1,
    frequency=0.038,
    amplitude=0.1,
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

    print(lipschitz_weighting, learning_rate, n_samples)
    print(parameters["lipschitz_weighting"], parameters["n_samples"], parameters["learning_rate"])

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   

    pdr = lipschitz.Lipschitz(model, optimizer, device, parameters)  

    # Calculation of the forget set importances
    #sample_importances = pdr.calc_importance(forget_train_dl)    
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

def emmn(
    model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    batch_size=4,
    **kwargs,
):

    noise_batch_size = batch_size
    retain_valid_dl = DataLoader(retain_valid_dl.dataset, batch_size=noise_batch_size)
    # collect some samples from each class

    forget_class_label = forget_class
    img_shape = next(iter(retain_train_dl.dataset))[0].shape[-1]
    noise = UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
    noise = UNSIR_noise_train(
        noise, model, forget_class_label, 25, noise_batch_size, device=device
    )

    retain_noises = {i_: UNSIR_noise(noise_batch_size, 3, img_shape, img_shape).to(device)
                     for i_ in [j for j in range(num_classes) if j!=forget_class]}
    for i_, n_ in retain_noises.items():
        retain_noises[i_] = emmn_noise_train(n_, model, i_, 30, noise_batch_size, device=device)

    noises = retain_noises
    noises[forget_class] = noise

    noisy_loader = emmc_create_noisy_lodaer(
        noises,
        batch_size=noise_batch_size,
        device=device,
    )        
    
    # impair step
    _ = fit_one_unlearning_cycle(
        2, model, noisy_loader, retain_valid_dl, device=device, lr=0.0001
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

# https://github.com/ayushkumartarun/zero-shot-unlearning 
def gkt(model,
    unlearning_teacher,
    retain_train_dl,
    retain_valid_dl,
    forget_train_dl,
    forget_valid_dl,
    valid_dl,
    num_classes,
    forget_class,
    device,
    batch_size=4,
    **kwargs,):
    
    if isinstance(model, ViT):
        batch_size = 8
    else:
        batch_size = 128
    n_generator_iter = 1 if isinstance(model, ViT) else 2
    n_student_iter = 5 if isinstance(model, ViT) else 15
    n_repeat_batch = n_generator_iter + n_student_iter
    
    idx_pseudo = 0
    total_n_pseudo_batches = 101 if isinstance(model, ViT) else 1000
    n_pseudo_batches = 0
    threshold = 0.01
    KL_temperature = 1
    AT_beta = 5


    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)   

    
    img_shape = next(iter(deepcopy(forget_valid_dl.dataset)))[0].shape[-1]    
    z_dim = 128 #they use 128 dim for 32x32 images, just adapting to that
    student = deepcopy(model)
    student = student.apply(weights_init)
    generator = LearnableLoader(n_repeat_batch=n_repeat_batch, batch_size=batch_size,z_dim=z_dim, out_size=img_shape, num_channels = 3, device = device).to(device=device)
    s_opt = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    g_opt = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.9)
    s_cycle = torch.optim.lr_scheduler.OneCycleLR(s_opt, max_lr=0.1, epochs=total_n_pseudo_batches, steps_per_epoch=n_student_iter)
    

    zero_break = 0
    zero_count = 0
    with tqdm(total=total_n_pseudo_batches) as pbar:
        gen_loss= []
        s_loss = []
        while n_pseudo_batches < total_n_pseudo_batches:            
            x_pseudo = generator.__next__()        
            preds, _ = model(x_pseudo)
            mask = (torch.softmax(preds.detach(), dim=1)[:, forget_class] <= threshold)
            #print(mask)
            x_pseudo = x_pseudo[mask]
            if x_pseudo.size(0) == 0:
                zero_count += 1
                if zero_count > 1000:
                    try:
                        print("Generator Stopped Producing datapoints corresponding to retain classes.")
                        print("Resetting the generator to previous checkpoint")
                        generator.load_state_dict(torch.load(str(((n_pseudo_batches//50)-1)*50) + ".pt"))
                    except Exception as e:
                        zero_break += 1
                        print(e)
                        if zero_break >= 50:
                            break
                continue
            else:
                zero_count = 0
                zero_break = 0            
            if idx_pseudo % n_repeat_batch < n_generator_iter:
                student_logits, student_activations = student(x_pseudo)
                teacher_logits, teacher_activations = model(x_pseudo)
                generator_total_loss = KT_loss_generator(student_logits, teacher_logits, KL_temperature=KL_temperature)
                gen_loss.append(generator_total_loss.item())
                g_opt.zero_grad()
                generator_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
                g_opt.step()
            elif idx_pseudo % n_repeat_batch < (n_generator_iter + n_student_iter):            
                
                with torch.no_grad():
                    teacher_logits, teacher_activations = model(x_pseudo)

                student_logits, student_activations = student(x_pseudo)
                student_total_loss = KT_loss_student(student_logits, student_activations, 
                                                    teacher_logits, teacher_activations, 
                                                    KL_temperature=KL_temperature, AT_beta = AT_beta)
                s_loss.append(student_total_loss.item())
                s_opt.zero_grad()
                student_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
                s_opt.step()   
                s_cycle.step()
            if (idx_pseudo + 1) % n_repeat_batch == 0: 
                # if n_pseudo_batches % 50 == 0:
                #     torch.save(generator.state_dict(), str(n_pseudo_batches) + ".pt")

                gen_loss = []
                s_loss = []
                n_pseudo_batches += 1
                 # saving the generator
                pbar.update(1)            
            idx_pseudo += 1    
    
    return get_metric_scores(
        student,
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


# From https://github.com/meghdadk/SCRUB

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
        nn.utils.clip_grad_norm_(model_s.parameters(), 1)
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

    print("IN FUNCTION")
    print(args.alpha, args.gamma)

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
    model_s = model
    
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
    
    for epoch in tqdm(range(1, args.sgda_epochs + 1)):

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

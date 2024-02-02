import copy
import torch.nn as nn
from torch.autograd import Variable
from typing import List
import numpy as np
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def train_distill(epoch, train_loader, module_list, model_s, criterion_list, optimizer, opt, split, quiet=False):
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
            input, target = data

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

        loss = loss + param_dist(model_s, model_s, opt.smoothing)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

    return
    
def unlearning(model, retain_loader, forget_loader):
    args = Namespace()
    args.optim = 'adam'
    args.gamma = 1
    args.alpha = 0.5
    args.beta = 0
    args.smoothing = 0.5
    args.msteps = 3
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 2
    args.distill = 'kd'

    """
    IMPORTANT: 
    ¬¬ sgda_epochs probably needs to be <=4 to not timeout (if not 2 or 3)
    ¬¬ sgda_decay_epochs will need to < sgda_epochs (obviously)
    ¬¬ 
    """
    args.sgda_epochs = 10
    args.sgda_learning_rate = 0.0005
    args.lr_decay_epochs = [5,8,9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 0.1#5e-4
    args.sgda_momentum = 0.9

    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)

    beta = 0.1
    def avg_fn(averaged_model_parameter, model_parameter, num_averaged): return (
        1 - beta) * averaged_model_parameter + beta * model_parameter
    
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
        optimizer = args.optim.SGD(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            momentum=args.sgda_momentum,
                            weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam": 
        optimizer = args.optim.Adam(trainable_list.parameters(),
                            lr=args.sgda_learning_rate,
                            weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = args    .optim.RMSprop(trainable_list.parameters(),
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
    
    acc_rs = []
    acc_fs = []
    acc_ts = []
    for epoch in range(1, args.sgda_epochs + 1):

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        # print("==> scrub unlearning ...")

        # acc_r, acc5_r, loss_r = validate(retain_loader, model_s, criterion_cls, args, True)
        # acc_f, acc5_f, loss_f = validate(forget_loader, model_s, criterion_cls, args, True)
        # acc_rs.append(100-acc_r.item())
        # acc_fs.append(100-acc_f.item())

        maximize_loss = 0
        if epoch <= args.msteps:
            train_distill(epoch, forget_loader, module_list, model_s, criterion_list, optimizer, args, "maximize")
        train_distill(epoch, retain_loader, module_list, model_s, criterion_list, optimizer, args, "minimize",)

        #print ("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss, train_loss, train_acc))

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn, optim
from torch import tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import *
from fed_cifar100 import load_partition_data_federated_cifar100
import random
import csv
import os
import time
import math
import torchvision.transforms.functional as TF
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        #if self.transform:
        #    image = self.transform(image)
        

        return image, label


class DatasetSplit_tensor(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]

        return image, torch.tensor(label)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def client_loader(dataset, user_groups, args):
    client_loader_dict = dict()
    #if args.model == 'resnet_sc':
    #    for client_idx, idxs_list in user_groups.items():
    #        client_loader_dict[client_idx] = DataLoader(DatasetSplit(
    #            dataset, list(idxs_list)), batch_size=args.local_bs, shuffle=True)
    #else:
    for client_idx, idxs_list in user_groups.items():
                client_loader_dict[client_idx] = DataLoader(DatasetSplit_tensor(
                    dataset, list(idxs_list)), batch_size=args.local_bs, shuffle=True,num_workers=4,pin_memory=True)
    return client_loader_dict


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    real_dataset = None
    test_dataset = None

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.dataset == 'cifar10':
            data_dir = '../data/cifar/'

        elif args.dataset == 'cifar100':
            data_dir = '../data/cifar100/'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        transforms_real = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    
        transforms_train = transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transforms_test = transforms.Compose([
            #transforms.CenterCrop(24),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        #A_transform = A.Compose([A.RandomCrop(24,24),ToTensorV2(),A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

        if args.dataset == 'cifar10' and args.model == 'resnet_sc':
            train_dataset = datasets.CIFAR10(data_dir,
                                         transform=TwoCropTransform(contrastive_transform),
                                         download=True)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms_test)
            real_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms_real)

        elif args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms_train)
            test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transforms_test)
            real_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transforms_real)

        elif args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transforms_train)
            test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transforms_test)
            real_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=transforms_real)

        
        """
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
        """
        client_class_set = None
        # sample training data amongst users
        if args.iid == 1:
            # Sample IID user data from Mnist
            # user_groups = cifar_iid(train_dataset, args.num_users)
            user_groups = cifar10_iid(train_dataset, args.num_users, args=args)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                if args.iid == 2:
#                    user_groups = partition_data(train_dataset, 'noniid-#label2', num_uers=args.num_users, alpha=1, args=args)
                    user_groups, client_class_set = cifar_noniid(train_dataset, num_users=args.num_users, args=args, class_num=1)
                else:
                    user_groups = partition_data(train_dataset, 'dirichlet', num_uers=args.num_users, alpha=args.alpha, args=args)
        # 분류된 index와 train dataset로 client train dataloder 생성
        client_loader_dict = client_loader(train_dataset, user_groups, args)
    elif args.dataset == 'fedcifar100':
        data_dir = '../data/fed_cifar100'
        train_dataset, test_dataset, client_loader_dict = load_partition_data_federated_cifar100(args=args, data_dir=data_dir, batch_size=args.local_bs)
        transforms_real = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

        real_dataset = datasets.CIFAR100('../data/cifar100/', train=True, download=True, transform=transforms_real)
        test_dataset = datasets.CIFAR100('../data/cifar100/', train=False, download=True, transform=transforms_real)
    elif args.dataset == 'tiny_imagenet':
        train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()])

        valid_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = datasets.ImageFolder('../data/tiny-imagenet-200/train', transform=train_transform)

        test_dataset = datasets.ImageFolder('../data/tiny-imagenet-200/val', transform=valid_transform)
        real_dataset = datasets.ImageFolder('../data/tiny-imagenet-200/train', transform=valid_transform)
        if args.iid == 1:
            # Sample IID user data from Mnist
            # user_groups = cifar_iid(train_dataset, args.num_users)
            user_groups = cifar10_iid(train_dataset, args.num_users, args=args)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                if args.iid == 2:
#                    user_groups = partition_data(train_dataset, 'noniid-#label2', num_uers=args.num_users, alpha=1, args=args)
                    user_groups, client_class_set  = cifar_noniid(train_dataset, num_users=args.num_users, args=args, class_num=4)
                else:
                    user_groups = partition_data(train_dataset, 'dirichlet', num_uers=args.num_users, alpha=args.alpha, args=args)                
        client_loader_dict = client_loader(train_dataset, user_groups, args)
        client_class_set = None
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.RandomCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std),
                                     ])

        valid_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std),
                                            ])

        train_dataset = datasets.ImageFolder("/mnt/disk2/workspace/Datasets/imagenet/train/", transform=train_transform)

        test_dataset = datasets.ImageFolder("/mnt/disk2/workspace/Datasets/imagenet/val/", transform=valid_transform)
        real_dataset = train_dataset
        if args.iid == 1:
            # Sample IID user data from Mnist
            # user_groups = cifar_iid(train_dataset, args.num_users)
            user_groups = cifar10_iid(train_dataset, args.num_users, args=args)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                if args.iid == 2:
#                    user_groups = partition_data(train_dataset, 'noniid-#label2', num_uers=args.num_users, alpha=1, args=args)
                    user_groups, client_class_set = imagenet_noniid(train_dataset, num_users=args.num_users, args=args, class_num=5)
                else:
                    user_groups = partition_data(train_dataset, 'dirichlet', num_uers=args.num_users, alpha=args.alpha, args=args)                
        client_loader_dict = client_loader(train_dataset, user_groups, args)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

        # 분류된 index와 train dataset로 client train data loader 생성
        client_loader_dict = client_loader(train_dataset, user_groups, args)

    return train_dataset, test_dataset, client_loader_dict,real_dataset,client_class_set


def average_weights_uniform(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_rebatch(w, client_loader_dict, idxs_users,batch_square):
    """
    Returns the average of the weights.
    
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]

    with torch.no_grad():
        #init
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                w_avg[key].data.copy_(w[0][key])
            else:
                w_avg[key] = w[0][key] * data_driven_ratio[0]
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key] * data_driven_ratio[i]
    return w_avg
    """
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]

    #init
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] *  data_driven_ratio[0] *1.0 

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] *  data_driven_ratio[i] *1.0 

    bn_idx = 0
    for key in w_avg.keys():
        if 'running_mean' in key:
            mean_square = w_avg[key]**2
        elif 'running_var' in key:
            w_avg[key] = (batch_square[bn_idx]- mean_square)
            bn_idx+=1
    return w_avg


def average_weights(w, client_loader_dict, idxs_users):
    """
    Returns the average of the weights.
    
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]

    with torch.no_grad():
        #init
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                w_avg[key].data.copy_(w[0][key])
            else:
                w_avg[key] = w[0][key] * data_driven_ratio[0]
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key] * data_driven_ratio[i]
    return w_avg
    """
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]

    #init
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] *  data_driven_ratio[0] *1.0 

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] *  data_driven_ratio[i] *1.0 
    return w_avg    
 
def average_weights_entropy(w, client_loader_dict, idxs_users,label_entropy):
    """
    Returns the average of the weights.
    
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]

    with torch.no_grad():
        #init
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                w_avg[key].data.copy_(w[0][key])
            else:
                w_avg[key] = w[0][key] * data_driven_ratio[0]
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key] * data_driven_ratio[i]
    return w_avg
    """
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]
    total_label_ratio = sum(label_entropy)
    label_entropy_ratio = [r/(total_label_ratio+ 0.000001) for r in label_entropy]
    #init
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * ( data_driven_ratio[0] *0.5 + label_entropy_ratio[0]*0.5)

    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * ( data_driven_ratio[i] *0.5 + label_entropy_ratio[i]*0.5)
    return w_avg


def average_weights_dyn(w, client_loader_dict, idxs_users, global_weights,num_clients,h):
        num_participants = len(w)
        alpha =0.001
        sum_theta = copy.deepcopy(w[0])
        for key in sum_theta.keys():
            sum_theta[key] = sum_theta[key] 

        for key in sum_theta.keys():
            for i in range(1, len(w)):
                sum_theta[key] += w[i][key] 
        delta_theta = {}
        for key in global_weights.keys():
            delta_theta[key] = sum_theta[key] - global_weights[key]

        for key in h.keys():
            h[key] -= alpha * (1./num_clients) * delta_theta[key]

        for key in global_weights.keys():
           global_weights[key] = (1./num_participants) * sum_theta[key] - (1./alpha) *  h[key]

        return global_weights,h

def average_weights_class_normalize(w, client_loader_dict,idxs_users, indices_class):
    """
    Returns the average of the weights.
    
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]

    with torch.no_grad():
        #init
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            if 'num_batches_tracked' in key:
                w_avg[key].data.copy_(w[0][key])
            else:
                w_avg[key] = w[0][key] * data_driven_ratio[0]
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key] * data_driven_ratio[i]
    return w_avg
    """
    total_data_points = sum([len(client_loader_dict[r].dataset) for r in idxs_users])
    data_driven_ratio = [len(client_loader_dict[r].dataset)/total_data_points for r in idxs_users]
    
    class_total_data_points = torch.sum(indices_class,dim=1)
    class_data_driven_ratio = torch.div(indices_class,class_total_data_points) 
    #init
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        if key == 'classifier.weight' :
            w_avg[key] = w_avg[key] * class_data_driven_ratio[0].reshape(10,1)
        elif key == 'classifier.bias':
            w_avg[key] = w_avg[key] * class_data_driven_ratio[0]
        else:
            w_avg[key] = w_avg[key] * data_driven_ratio[0]

    for key in w_avg.keys():
        for i in range(1, len(w)):
            if key == 'classifier.weight' :
                w_avg[key] += w[i][key] * class_data_driven_ratio[i].reshape(10,1)
            elif key == 'classifier.bias':
                w_avg[key] += w[i][key] * class_data_driven_ratio[i]
            else:
                w_avg[key] += w[i][key] * data_driven_ratio[i]
    return w_avg

def set_running_statics(w_avg, global_model, server_momentum):
    global_model_state_dict = global_model.state_dict()
    delta_static = copy.deepcopy(w_avg)
    for key in w_avg.keys():
        if 'running' in key:
            delta_static[key] = global_model_state_dict[key] - w_avg[key]
            server_momentum[key] = 0.9 * server_momentum[key] + delta_static[key]
            global_model_state_dict[key] = global_model_state_dict[key] - server_momentum[key]

           #simple avg
           #global_model_state_dict[key] = w_avg[key]
    global_model.load_state_dict(global_model_state_dict)

def set_model_global_grads(w_avg, global_model):
    # upload update param
    new_model = copy.deepcopy(global_model)
    new_model.load_state_dict(w_avg)

    with torch.no_grad():
        for parameter, new_parameter in zip(
                global_model.parameters(), new_model.parameters()
        ):  
            # between last model & update model
            if parameter.requires_grad:
                parameter.grad = parameter.data - new_parameter.data
            else:
                print('check')
    #new_model_state_dict = new_model.state_dict()
    #for k in dict(global_model.named_parameters()).keys():
    #    new_model_state_dict[k] = global_model_state_dict[k]
   
    #print(global_model.fc3.weight.grad.clone().detach())
    bn_idx = 0
    return global_model

def set_model_global_grads_qffl(w, global_model, lr, loss, q):
    # upload update param
    # loss가 idx가 아니라 모든 data loss인듯? all client loss??
    #idx의 경우 loss -> loss[idx]
    new_model = copy.deepcopy(global_model)
    
    Deltas = []
    hs=[]
    with torch.no_grad():
        for idx, w_ in enumerate(w):
            grads = []
            new_model.load_state_dict(w_)
            for parameter, new_parameter in zip(
                    global_model.parameters(), new_model.parameters()
            ):  
                # between last model & update model
                if parameter.requires_grad:
                    grads.append((parameter.data - new_parameter.data) * 1.0 / lr)
                else:
                    print('check')
            Deltas.append([torch.float_power(loss[idx]+1e-10, q) * grad for grad in grads])
            hs.append(q * torch.float_power(loss[idx]+1e-10, (q-1)) * norm_grad(grads) + (1.0/lr) * torch.float_power(loss[idx]+1e-10,q))
        updates = aggregate(global_model,Deltas,hs)
    #updates가 이미 global - w_avg임
    #즉 global - updates = w_avg
    for idx, param in enumerate(global_model.parameters()):
        param.grad = updates[idx]
    #global_model = set_model_global_grads(new_model.state_dict(),global_model)
    return global_model

def norm_grad(grad_list):
    # input: nested gradients
    # output: square of the L-2 norm

    client_grads = torch.flatten(grad_list[0])

    for i in range(1, len(grad_list)):
        client_grads = torch.cat((client_grads,torch.flatten(grad_list[i]))) # output a flattened array  

    return torch.sum(torch.square(client_grads))

def aggregate(weights_before, Deltas, hs): 
    
    #전체 h
    demominator = np.sum(np.asarray(hs))
    num_clients = len(Deltas)
    scaled_deltas = []
    for client_delta in Deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])

    updates = []
    #for layer
    # client 0 init -> for client 1 ~ len deltas
    for i in range(len(Deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(Deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)

    

    return updates

def qffl_infernece(args, test_model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """
    model = copy.deepcopy(test_model)

   # model.to(device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()

    testloader = DataLoader(test_dataset, batch_size=100,
                            shuffle=False)
    # cifar 10 기준 50번(100*50)이 한번의 class
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            #print(outputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            class_correct  = torch.sum(torch.eq(pred_labels, labels)).item()
            correct += class_correct
            total += len(labels)
            
  

        
    #for i in range(10): class_acc[i]/=1000 
    del images
    del labels
    #print(class_acc)
    return loss
def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x ** 2, dim=dim, keepdim=keepdim) ** 0.5


def get_logger(file_path):
    logging.basicConfig(level=logging.INFO,
                        # logging.basicConfig(level=logging.DEBUG,
                        format=' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    logger = logging.getLogger()
    #log_format = '%(asctime)s | %(message)s'
    #formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    #file_handler.setFormatter(formatter)
    #stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    #logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def check_norm(model):
    total_norm = 0
    for name, p in model.named_parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
#        print('name:{}, gradient norm{}'.format(name, param_norm))
    total_norm = total_norm ** (1. / 2)
    #same
#    print('conv norm:{}'.format(model.conv1.weight.grad.clone().norm(p=2)))
#    print('gn norm{}'.format(model.bn1.weight.grad.clone().norm(p=2)))
#    print('conv2 norm{}'.format(model.conv2.weight.grad.clone().norm(p=2)))
#    print('fc norm{}'.format(model.fc.weight.grad.clone().norm(p=2)))
#    print('total_norm: {}'.format(total_norm))
    return total_norm


class CSVLogger(object):
    def __init__(self, filename, keys):
        self.filename = filename
        self.keys = keys
        self.values = {k: [] for k in keys}
        self.init_file()

    def init_file(self):
        # This will overwrite previous file
        if os.path.exists(self.filename):
            return

        directory = os.path.dirname(self.filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.filename, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(self.keys)

    def write_row(self, values):
        assert len(values) == len(self.keys)
        if not os.path.exists(self.filename):
            self.init_file()
        with open(self.filename, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(values)


def exp_decay_schedule_builder(base_value, decay_steps, decay_rate, staircase, round_num):
  """Creates a learning rate schedule with exponential root decay.

  Args:
    base_value: The base value of the quantity to decay over time.
    decay_steps: A positive scalar that governs how much the value decays at a
      given round number.
    decay_rate: A float between 0 and 1 that governs how quickly the decay
      occurs.
    staircase: A boolean. If set to True, the decaying occurs in discrete
      intervals.

  Returns:
    A 1-arg callable that produces a decayed version of the base value when
      passed the current round_num.
  """
  if staircase:
    def exp_decay_fn(round_num):
      return base_value * pow(decay_rate, round_num // decay_steps)
  else:
    def exp_decay_fn(round_num):
      return base_value * pow(decay_rate, round_num / decay_steps)

  return exp_decay_fn(round_num)




def eval_smooth(prev_model, model, num_pts=1):
 
    alphas = np.arange(1, num_pts+1)/(num_pts+1)
    gnorm = check_norm(prev_model)
    update_size = norm_diff(get_model_params(model), \
                                  get_model_params(prev_model))
    max_smooth = -1
    for alpha in alphas:
        new_model = copy.deepcopy(prev_model)
        
        for n, p in new_model.named_parameters():
            p.data = alpha * p.data + (1-alpha) * {n:p for n, p in model.named_parameters()}[n].data
            
        check_norm(new_model)
        smooth = norm_diff(utils.get_model_grads(new_model), get_model_grads(prev_model))/ (update_size * (1- alpha))
        max_smooth = max(smooth, max_smooth)
    
    return max_smooth, gnorm

def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def average(opt, gmf):
   # weight (float): relative sample size of client
    weight = 0.1
    param_list = []
    # 하나의 클라이언트? param_groups -> lr, weight_decay, momentum, dampening 등
    for group in opt.param_groups:
        for p in group['params']:
            param_state = opt.state[p]
            param_state['cum_grad'].mul_(weight)
            param_list.aapend(param_state['cum_grad'])
    
    #flat_tensor = flatten_tensors(param_list, w)
    #communication_op(tensor=flat_tensor)
    #for f, t in zip(unflatten_tensors(flat_tensor, param_list), param_list):
    #    t.set_(f)
    
    for group in opt.param_groups:
        lr = group['lr']
        for p in group['params']:
                param_state = opt.state[p]

                if gmf != 0:
                    if 'global_momentum_buffer' not in param_state:
                        buf = param_state['global_momentum_buffer'] = torch.clone(param_state['cum_grad']).detach()
                        buf.div_(lr)
                    else:
                        buf = param_state['global_momentum_buffer']
                        buf.mul_(gmf).add_(1/lr, param_state['cum_grad'])
                    param_state['old_init'].sub_(lr, buf)
                else:
                    param_state['old_init'].sub_(param_state['cum_grad'])
                
                p.data.copy_(param_state['old_init'])
                param_state['cum_grad'].zero_()

                # Reinitialize momentum buffer
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()
        

def get_model_grads(model):
    return [p.grad.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]

def get_model_params(model):
    return [p.data for _, p in model.named_parameters() if \
            hasattr(p, 'grad') and (p.grad is not None)]


def norm_diff(list1, list2=None):
    if not list2:
        list2 = [0] * len(list1)
    assert len(list1) == len(list2)
    return math.sqrt(sum((list1[i]-list2[i]).norm()**2 for i in range(len(list1))))


def cifar10_transform(train = True):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api

    transoformed_img = torch.stack([cifar10_transform
        (train)
        (i)
        for i in img])
    return transoformed_img  

def augment(images_real, param_augment):
    # This can be sped up in the future.

    if param_augment != None and param_augment['strategy'] != 'none':

        strategy = param_augment['strategy']

        shape = images_real.shape
        
        augs = strategy.split('_')
        idx =1
        for i in range(idx):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            images = images_real
            if choice == 'crop':
                #rand crop
                images = TF.pad(images,4)
                i = random.randint(0, 4)
                j = random.randint(0, 4)
                images = TF.crop(images,i,j,i+32,j+32)
            elif choice == 'gray':
                images = TF.rgb_to_grayscale(images,3)
            elif choice == 'hflip':
                images = TF.hflip(images)
            elif choice == 'brightness':
                rand = random.randint(2,30)
                images = TF.adjust_brightness(images,rand*0.1)
            elif choice == 'contrast':
                rand = random.randint(2,30)
                images = TF.adjust_contrast(images,rand*0.1)
            elif choice == 'saturation':
                rand = random.randint(2,30)
                images = TF.adjust_saturation(images,rand*0.1)
            elif choice == 'hue':
                rand = random.randint(0,100)
                rand -=50
                rand *= 0.01
                images = TF.adjust_hue(images,rand*0.1)
        
    return TF.normalize(images_real,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def augment_batch(images_real, param_augment):
    # This can be sped up in the future.

    if param_augment != None and param_augment['strategy'] != 'none':

        strategy = param_augment['strategy']

        shape = images_real.shape
        
        augs = strategy.split('_')
        idx =1
        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            images = images_real[i,:,:,:]
            if choice == 'crop':
                #rand crop
                images = TF.pad(images,4)
                i = random.randint(0, 4)
                j = random.randint(0, 4)
                images = TF.crop(images,i,j,i+32,j+32)
            elif choice == 'gray':
                images = TF.rgb_to_grayscale(images,3)
            elif choice == 'hflip':
                images = TF.hflip(images)
            elif choice == 'brightness':
                rand = random.randint(2,30)
                images = TF.adjust_brightness(images,rand*0.1)
            elif choice == 'contrast':
                rand = random.randint(2,30)
                images = TF.adjust_contrast(images,rand*0.1)
            elif choice == 'saturation':
                rand = random.randint(2,30)
                images = TF.adjust_saturation(images,rand*0.1)
            elif choice == 'hue':
                rand = random.randint(0,100)
                rand -=50
                rand *= 0.01
                images = TF.adjust_hue(images,rand*0.1)
    return TF.normalize(images_real,(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


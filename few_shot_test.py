import asyncio
import websockets;
import pickle

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torchvision.models
from networks import ConvNet
from torchvision import datasets, transforms
import random
from mobilnet import mobilenet_v2
from models import ImageNet_ResNet
import nest_asyncio
nest_asyncio.apply()
#__import__('IPython').embed()
import pickle
import logging
import colorlog
from typing import List
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
#CLASSIFIER_WEIGHT = "group2.fc.weight"
#CLASSIFIER_BIAS = "group2.fc.bias"
#CLASSIFIER = "group2.fc"
CLASSIFIER_WEIGHT = "classifier.1.weight"
CLASSIFIER_BIAS = "classifier.1.bias"
CLASSIFIER = "classifier.1"


class tsnehook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        #nch = input[0].shape[1]

        #mean = input[0].mean([0, 2, 3])
        #var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        #r_feature = torch.norm(self.var - var, 2) + torch.norm(
        #    self.mean - mean, 2)


        self.output = input
        # must have no output

    def close(self):
        self.hook.remove()


def pickle_dict(user, state_dict, unseen):
    return pickle.dumps({'user':user,'state_dict':state_dict,'unseen':unseen})
def init_logger(dunder_name) -> logging.Logger:
    log_format = (
        '[%(asctime)19s - '
        '%(name)s - '
        '%(levelname)s]  '
        '%(message)s'
    )
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    colorlog.basicConfig(format=colorlog_format)
    logger = logging.getLogger(dunder_name)

    logger.setLevel(logging.INFO)

    # Output full log
    fh = logging.FileHandler('app.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def weight_normalize(weight):
    norm = torch.norm(weight,2,1,keepdim=True)
    weight = weight / torch.pow(norm,1)
    return weight
def weight_align(base_weight):

    mean_base = torch.mean(torch.norm(base_weight,dim=1)).item()
    mean_novel = torch.norm(base_weight,dim=1)
    gamma = mean_novel / mean_base
    base_weight = torch.mul(base_weight.T,gamma)
    print("mean_base:{}, mean_novel:{}".format(mean_base,mean_novel))
    return base_weight.T

def surgery_(model:torch.nn.Module, num_unseen:int):
    
    weight = copy.deepcopy(model.state_dict()[CLASSIFIER_WEIGHT])
    bias = copy.deepcopy(model.state_dict()[CLASSIFIER_BIAS])
    #mobilenet
    model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(weight.shape[1], 1005),
    )
    #model.group2 = nn.Sequential(
    #    OrderedDict([
    #        ('fc', nn.Linear(weight.shape[1], 1000))
    #    ])
   # )

    unseen_weights = torch.empty(num_unseen,weight.shape[1], device="cuda:0")
    torch.nn.init.xavier_normal_(unseen_weights)
    
    unseen_bias = torch.randn(num_unseen, device="cuda:0")
    #for idx,param in enumerate(model.classifier.parameters()):
    #    if idx==0:
    #        param.data= torch.cat([weight, unseen_weights], dim=0)
    #    else:
    #        param.data = torch.cat([bias, unseen_bias], dim=0)
    
    dict = model.state_dict() 
    dict[CLASSIFIER_WEIGHT][:1000,:] = weight
    dict[CLASSIFIER_BIAS][:1000] = bias
    dict[CLASSIFIER_WEIGHT][1000:,:] = unseen_weights
    dict[CLASSIFIER_BIAS][1000:] = unseen_bias
    
    model.load_state_dict(dict)
    model.to(device)
 
def surgery(model:torch.nn.Module, num_unseen:int, name:str = "classifier.1"):
    for n, module in model.named_modules():
        if n == name:
            m = module

    if m.weight != None:
        #TODO device to config
        unseen_weights = torch.empty(num_unseen, m.weight.shape[1], device="cuda:0")
        torch.nn.init.xavier_normal_(unseen_weights)
        tmp = torch.cat([m.weight, unseen_weights], dim=0)
        m.weight.data = tmp

    if m.bias != None:
        unseen_weights = torch.randn(num_unseen, device="cuda:0")
        #torch.nn.init.xavier_normal(unseen_weights)
        tmp = torch.cat([m.bias, unseen_weights], dim=0)
        m.bias.data = tmp


    

def inference(model, test_dataset, device):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100,
                                             shuffle=False)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            outputs = model(images)
            # print(outputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            class_correct = torch.sum(torch.eq(pred_labels, labels)).item()
            correct += class_correct
            total += len(labels)
    # for i in range(10): class_acc[i]/=1000
    accuracy = correct / len(test_dataset)
    del images
    del labels
    # print(class_acc)
    return accuracy, loss / len(iter(testloader))


def get_fewshot_dataset(root: str, trasnform: transforms, class_to_idx, shot=10, base=True,fewshot=True):
    dataset = datasets.ImageFolder(root, transform=trasnform)
    data, labels = dataset.imgs, dataset.targets
    indices_class = [[] for _ in range(len(dataset.classes))]

    for i, lab in enumerate(labels):
        indices_class[lab].append(i)

    origin_labels = []
    if base:
        for key in dataset.class_to_idx:
            origin_labels.append(class_to_idx[key])

        for idx, label in enumerate(labels):
            labels[idx] = origin_labels[label]
    else:
        for idx, label in enumerate(labels):
            labels[idx] += 1000

    """10-shot dataset 만들기"""
    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        # idx_shuffle = indices_class[c][:n]
        return idx_shuffle

    all_class_idx = list()

    if fewshot==True:
        for c in range(len(dataset.classes)):
            all_class_idx.append(get_images(c, shot))  # 10 shot
        all_class_idx = np.concatenate(all_class_idx)
    else:
        for c in range(len(dataset.classes)):
            all_class_idx.append(indices_class[c])  # 10 shot
        all_class_idx = np.concatenate(all_class_idx)

    data_list = list()
    for class_idx in all_class_idx:
        data_list.append((data[class_idx][0], labels[class_idx]))

    dataset.imgs = data_list
    dataset.samples = data_list
    dataset.targets = list(np.array(labels)[all_class_idx])
    return dataset


def seed_fix(logger, seed=0):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"seed: {seed}")

def cut_classifier_dict(state_dict, classes):
    extend_weight = state_dict[CLASSIFIER_WEIGHT][classes:,:]
    extend_bias = state_dict[CLASSIFIER_BIAS][classes:]
    state_dict[CLASSIFIER_WEIGHT] = state_dict[CLASSIFIER_WEIGHT][:classes,:]
    state_dict[CLASSIFIER_BIAS] = state_dict[CLASSIFIER_BIAS][:classes]
    
    #print("Base shape:W {},b {}, Novel shape:W {},b {}".format(state_dict[CLASSIFER_WEIGHT].shape, state_dict[CLASSIFER_BIAS].shape, extend_weight.shape, extend_bias.shape)) 
    return copy.deepcopy(extend_weight), copy.deepcopy(extend_bias), copy.deepcopy(state_dict)

def few_shot_task(model):
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

    writer = SummaryWriter('runs/few')
    """Imagenet """
    # key = class name, value = index
    with open('class_to_idx.pickle', 'rb') as f:
        imagenet_class_to_idx = pickle.load(f)
    train_dataset = get_fewshot_dataset("/home/seongwoongkim/Projects/data/jetson1_imagenet_10/train", train_transform,imagenet_class_to_idx,fewshot=False)
    valid_set = get_fewshot_dataset("/home/seongwoongkim/Projects/data/ILSVRC2012_img_validation/", valid_transform,imagenet_class_to_idx,fewshot=False)
    origin_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=64, shuffle=True, num_workers=4, pin_memory=True)#, collate_fn=lambda x:x)
    optimizer_origin = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=4e-5)

    # 10 classes corresponding to a single clint
    base_dataset = get_fewshot_dataset("/home/seongwoongkim/Projects/data/jetson1_imagenet_10/train", train_transform, imagenet_class_to_idx)
    test_dataset = get_fewshot_dataset("/home/seongwoongkim/Projects/data/jetson1_imagenet_10/validation", valid_transform, imagenet_class_to_idx,fewshot=False)
    logger.info(f"base_class: {base_dataset.classes}")

    """Novel Class """
    novel_dataset = get_fewshot_dataset("/home/seongwoongkim/Projects/data/Novel_two_dataset/train", train_transform, imagenet_class_to_idx, base=False)
    novel_test_dataset = get_fewshot_dataset("/home/seongwoongkim/Projects/data/Novel_two_dataset/validation", valid_transform, imagenet_class_to_idx, base=False,fewshot=False)
    logger.info(f"num novel classes: {len(novel_dataset.classes)}")
    logger.info(f"novel train dataset size:  {len(novel_dataset.imgs)}")
    # novel_class_idx 0->1000, 1->1001
    concated_dataset = torch.utils.data.ConcatDataset([base_dataset, novel_dataset])
    train_loader = torch.utils.data.DataLoader(concated_dataset,
                                            batch_size=32, shuffle=True, num_workers=4, pin_memory=True)#, collate_fn=lambda x:x)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=4e-5)
    criterion = nn.CrossEntropyLoss()
    novel_train_dataset = get_fewshot_dataset("/home/seongwoongkim/Projects/data/Novel_one_dataset/train", valid_transform, imagenet_class_to_idx, base=False)

    '''novel_loader = torch.utils.data.DataLoader(novel_train_dataset,
                                            batch_size=50, shuffle=False, num_workers=4, pin_memory=True)
    features =[]
    for module in model.modules():
        if module.__class__.__name__ == 'Linear':
            features = tsnehook(module)
    with torch.no_grad():
        model.eval()
        for batch_idx, (images, labels) in enumerate(novel_loader):
            images.requires_grad = True
            labels.requires_grad = False
            images, labels = images.to(device), labels.to(device)

            optimizer_origin.zero_grad()

            output = model(images)
    novel_featreus = copy.deepcopy(features.output[0])
    del features'''
    model.train()
    epochs = 20
    unseen = True
    global_round = 10

    '''for name, param in model.named_parameters():
        if CLASSIFIER in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    #for name, param in model.named_parameters():
    #    print(param.requires_grad)
    print()
    logger.info(f"training begin...")
    print("Base bias:{}".format(torch.mean(model.state_dict()[CLASSIFIER_BIAS])))
    print("Base weight:{}".format(torch.mean(torch.norm(model.state_dict()[CLASSIFIER_WEIGHT][:1000,:],dim=1)).item()))

    for batch_idx, (images, labels) in enumerate(origin_loader):
        images.requires_grad = True
        labels.requires_grad = False
        images, labels = images.to(device), labels.to(device)

        optimizer_origin.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        g_loss = loss.item()

        loss.backward()

        optimizer_origin.step()

        torch.cuda.empty_cache()'''

    accuracy, loss = inference(model, test_dataset, device=device)
    print('Test Base Accuracy: {:.2f}%'.format(100 * accuracy))
    print(f'Test Base Loss: {loss} \n')
    save_origin = copy.deepcopy(model.state_dict())
    for round in range(global_round):
        print("Participating in {} round \n".format(round+1))

        model.train()
    
        if unseen==True:
            surgery(model, len(novel_dataset.classes))
            logger.info(f"surgery done")
            print("Base bias:{}, Novel bias:{}".format(torch.mean(model.state_dict()[CLASSIFIER_BIAS][:1000]).item(),torch.mean(model.state_dict()[CLASSIFIER_BIAS][1000:]).item()))
            print("Base weight:{}, Novel weight:{}".format(torch.mean(torch.norm(model.state_dict()[CLASSIFIER_WEIGHT][:1000,:],dim=1)).item(),torch.mean(torch.norm(model.state_dict()[CLASSIFIER_WEIGHT][1000:,:],dim=1)).item()))
        for name, param in model.named_parameters():
            if CLASSIFIER in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        #for name, param in model.named_parameters():
        #    print(param.requires_grad)
        #con_freeze(model)
        logger.info(f"freeze done")


        for epoch in range(epochs):
            model.train()
            g_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                state_dict = model.state_dict()
                state_dict[CLASSIFIER_WEIGHT][1000:,:] = state_dict[CLASSIFIER_WEIGHT][1000:,:] - torch.mean(state_dict[CLASSIFIER_WEIGHT][1000:,:],dim=1).reshape(5,-1)
                state_dict[CLASSIFIER_BIAS][1000:] = torch.zeros_like(state_dict[CLASSIFIER_BIAS][1000:] ,device="cuda:0")
                state_dict[CLASSIFIER_WEIGHT][:1000,:] = state_dict[CLASSIFIER_WEIGHT][:1000,:] - torch.mean(state_dict[CLASSIFIER_WEIGHT][:1000,:],dim=1).reshape(1000,-1)
                model.load_state_dict(state_dict)

                images.requires_grad = True
                labels.requires_grad = False
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                output = model(images)
                loss = criterion(output, labels)
                g_loss = loss.item()

                loss.backward()

                optimizer.step()

                torch.cuda.empty_cache()
                '''for name,param in model.named_parameters():
                    if CLASSIFIER_WEIGHT == name:
                        base = param[0,:].clone().cpu().data.numpy()
                        novel = param[1000,:].clone().cpu().data.numpy()
                        writer.add_histogram(name+"base",base,batch_idx*(epoch+1))
                        writer.add_histogram(name+"novel",novel,batch_idx*(epoch+1))'''
                    #writer.add_histogram(name+"1",novel,epoch)
            logger.info(f"epoch {epoch}, loss {g_loss:.4f}")
            if (epoch+1) % 20 == 0:
                state_dict = model.state_dict()

                save_dict = copy.deepcopy(state_dict)
                state_dict = save_origin
                state_dict[CLASSIFIER_WEIGHT]= save_dict[CLASSIFIER_WEIGHT]
                state_dict[CLASSIFIER_BIAS]= save_dict[CLASSIFIER_BIAS]
                
                state_dict[CLASSIFIER_WEIGHT][1000:,:] = state_dict[CLASSIFIER_WEIGHT][1000:,:] - torch.mean(state_dict[CLASSIFIER_WEIGHT][1000:,:],dim=1).reshape(5,-1)
                state_dict[CLASSIFIER_BIAS] = torch.zeros_like(state_dict[CLASSIFIER_BIAS],device="cuda:0")
                state_dict[CLASSIFIER_WEIGHT][:1000,:] = state_dict[CLASSIFIER_WEIGHT][:1000,:] - torch.mean(state_dict[CLASSIFIER_WEIGHT][:1000,:],dim=1).reshape(1000,-1)

                state_dict[CLASSIFIER_WEIGHT] = weight_align( state_dict[CLASSIFIER_WEIGHT])
                
                #state_dict[CLASSIFIER_WEIGHT] = weight_normalize(state_dict[CLASSIFIER_WEIGHT])
                model.load_state_dict(state_dict)

                accuracy, loss = inference(model, valid_set, device=device)
                print('Test Base Accuracy: {:.2f}%'.format(100 * accuracy))
                print(f'Test Base Loss: {loss} \n')

                accuracy, loss = inference(model, novel_test_dataset, device=device)
                print('Test Novel Accuracy: {:.2f}%'.format(100 * accuracy))
                print(f'Test Novel Loss: {loss} \n')


                model.load_state_dict(save_dict)
        print("Base bias:{}, Novel bias:{}".format(torch.mean(model.state_dict()[CLASSIFIER_BIAS][:1000]).item(),torch.mean(model.state_dict()[CLASSIFIER_BIAS][1000:]).item()))
        print("Base weight:{}, Novel weight:{}".format(torch.mean(torch.norm(model.state_dict()[CLASSIFIER_WEIGHT][:1000,:],dim=1)).item(),torch.mean(torch.norm(model.state_dict()[CLASSIFIER_WEIGHT][1000:,:],dim=1)).item()))
        print(torch.mean(torch.mean(model.state_dict()[CLASSIFIER_WEIGHT][:1000,:],dim=1)))
        print(torch.mean(torch.mean(model.state_dict()[CLASSIFIER_WEIGHT][1000:,:],dim=1)))
        unseen= False
        #cut_classifier_dict(model.state_dict(),1000)
    return  model.state_dict()
if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn')
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    pruning_ratio = 0.1
    device = 'cuda:0'
    logger = init_logger(__name__)
    seed_fix(logger)

    logger.info(f"pruning ratio: {pruning_ratio}")

    """mobilenet"""
    model = mobilenet_v2(pretrained=False, pruning_ratio=1 - pruning_ratio, width_mult=0.75)
    m = torch.load("mobilenet_lbyl_0.1")
    model.load_state_dict(m['global_model'])
    
    """resnet"""
    
    '''cfg = [64, 128, 256, 512]
    for i in range(len(cfg)):
        cfg[i] = int(cfg[i] * (1 - pruning_ratio))
    temp_cfg = cfg
    model = ImageNet_ResNet.resnet34(pretrained= False, cfg=cfg)
    m = torch.load("resnet34_{}_lbyl".format(pruning_ratio))
    model.load_state_dict(m['global_model'])'''

    model.to(device)

    origin_state_dict = model.state_dict()
    few_shot_model = copy.deepcopy(model)
    state_dict = few_shot_task(model)
    extend_weight, extend_bias, state_dict = cut_classifier_dict(state_dict,classes=1000)
    # 서버모델에 합산
    surgery(model,5,extend_weight,extend_bias)
    
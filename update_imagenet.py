#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import torch
from torch import nn
from torch._C import dtype
from torch.nn import parameter
from torchvision import transforms
from sam import SAM
from sgd_agc import SGD_AGC
from agc import AGC
from torch.utils.data import DataLoader, Dataset
from uodate_features import CrossEntropy
from utils import check_norm, exp_decay_schedule_builder, eval_smooth, preprocess_cifar_img, augment_batch
from FedNova import *
from pc_grad import PCGrad
from pcgrad import pc_grad_update
import copy
import random
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import time
from resnet import ResNet32_test, ResNet32_nobn
from networks import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar_fedVC, CNNCifar_WS, AlexNet, ConvNet
import torchvision.transforms.functional as TF
from tsnecuda import TSNE
import seaborn as sns
import pandas as pd
from noise_loss import *
from FedNova import FedProxOptimizer, FedProx

CLASSIFIER_WEIGHT = "classifier.weight"
CLASSIFIER_BIAS = "classifier.bias"
CLASSIFIER = "classifier"

class BalancedSoftmax(torch.nn.Module):
    def __init__(self, device, num_classes=10):
        super(BalancedSoftmax, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()
    def forward(self, pred, labels, sample_per_class):
        # CCE
        spc = sample_per_class.type_as(pred)
        spc = spc.unsqueeze(0).expand(pred.shape[0], -1)
        logits = pred + spc.log()
        #pred = F.softmax(logits,dim=1)
        #pred = torch.clamp(pred, min=1e-7, max=1.0)

        loss = F.cross_entropy(input=logits, target=labels)
        '''pred = F.gumbel_softmax(logits,tau=0.2)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).double().to(self.device)
        #label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        loss = -1 * torch.sum(label_one_hot * torch.log(pred), dim=1) '''
        return loss


class CEandBS(torch.nn.Module):
    def __init__(self, device, num_classes,alpha=0.5, beta=0.5):
        super(CEandBS, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss()
        self.se = BalancedSoftmax(device=self.device, num_classes=num_classes)
        self.alpha = alpha
        self.beta = beta
    def forward(self, pred, labels, sample_per_class):
        return self.alpha * self.ce(pred, labels) + self.beta * self.se(pred, labels,sample_per_class)

class LocalUpdate(object):
    def __init__(self, args, train_loader, device):
        # idxs= client의 data index ex) 5000개 data의 index집합

        self.args = args
        self.trainloader = train_loader

        self.device = device
        #self.client_class = list(client_class)
        # Default criterion set to NLL loss function
       
        #elf.criterion = SCELoss(alpha=self.args.momentum,beta=self.args.dc_lr,device=device, num_classes=10)
        #self.criterion = CEandMAE(alpha=self.args.momentum,beta=self.args.dc_lr,device=device, num_classes=10)

        #self.criterion = NCEandRCE(alpha=self.args.momentum,beta=self.args.dc_lr,num_classes=10)
        #self.criterion = GCELoss(device=device,,num_classes=10)
        #self.criterion = torch.nn.CrossEntropyLoss()

    def update_weights(self, model, global_round, idx_user):
        # Set mode to train model
#        model.to(self.device)
        model.train()
        epoch_loss = []
        total_norm = []
        loss_list = []
        conv_grad = []
        fc_grad = []
        global_model = copy.deepcopy(model)
        self.global_model = global_model
        '''
        batch augamentation과정중 문제 발생. 해결필요-iteration 적용 등

        # 시드 고정
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.args.seed+ idx_user)
        torch.cuda.manual_seed(self.args.seed+ idx_user)
        random.seed(self.args.seed+ idx_user)
        np.random.seed(self.args.seed+ idx_user)
        '''
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=4e-5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        elif self.args.optimizer == 'no_momentum':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr,
                                        weight_decay=1e-3)
        elif self.args.optimizer == 'sam':
            base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
            optimizer = SAM(model.parameters(), base_optimizer, lr=self.args.lr, momentum=0.9, weight_decay=1e-4)
        elif self.args.optimizer == 'no_weight_decay':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr
                                        )
        elif self.args.optimizer == 'clip':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=1e-3
                                        )
        elif self.args.optimizer == 'fedprox':
            optimizer = FedProx(model.parameters(), lr=self.args.lr, weight_decay=1e-3, mu=0.1,ratio=0.05
                                        )
        elif self.args.optimizer == 'clip_nf':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=1e-4
                                        )
        
            if 'resnet18' in self.args.model:
                optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['linear'], clipping=1e-3)
            elif 'resnet' in self.args.model:
                optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc'], clipping=1e-3)
            else:
                optimizer = AGC(model.parameters(), optimizer, model=model, ignore_agc=['fc1', 'fc2', 'fc3'],
                                clipping=1e-3)
            # optimizer = SGD_AGC(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4, clipping=1e-3)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4
                                    )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
        origin_dataset = self.trainloader.dataset     
        labels_all = origin_dataset.dataset.targets
        indices_class = [[] for c in range(self.args.num_classes)]
        idxs = origin_dataset.idxs
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)
        labels_all = labels_all[idxs]
        for i, lab in enumerate(labels_all):
           indices_class[lab].append(i)
        indicate_class = []
        all_idx = 0
        for i in indices_class:
            all_idx+=len(i)
            indicate_class.append(len(i))           
        #client_class_idx = np.array([i for i in range(self.args.num_classes)])
        #client_class_idx = np.delete(client_class_idx,self.client_class)
        indicate_class = np.array(indicate_class)
        indicate_class = torch.tensor(indicate_class,device=self.device)
        if self.args.loss=='ce':
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = BalancedSoftmax(self.device,self.args.num_classes)
        #self.criterion = CEandBS(self.device, self.args.num_classes)

        for iter_ in range(self.args.local_ep):
            batch_loss = []
            loss_avg = 0
           
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                #torch.manual_seed(self.args.seed+ idx_user+batch_idx)
                #images = augment(images, param_augment)

                #                 #if batch_idx==0:
                #    print('idx_user: {}, label: {}'.format(idx_user, labels[0]))
                images.requires_grad=True
                labels.requires_grad=False
                optimizer.zero_grad()
                #state_dict = model.state_dict()
        
                #state_dict[CLASSIFIER_BIAS][:1000] = torch.zeros_like(state_dict[CLASSIFIER_BIAS][:1000] ,device="cuda:0")
                if self.args.kernel_sizes == 'centering':
                    for name, param in model.named_parameters():
                    #print(name)
                       if 'classifier.weight' in name:
                           param.data = param.data - torch.mean(param.data,dim=1).reshape(param.data.shape[0],-1)
                        #elif 'weight' in name:
                        #    weight_mean =  param.data.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                        #    param.data  =  param.data  - weight_mean
                #    state_dict[CLASSIFIER_WEIGHT] = state_dict[CLASSIFIER_WEIGHT] - torch.mean(state_dict[CLASSIFIER_WEIGHT],dim=1).reshape(state_dict[CLASSIFIER_WEIGHT].shape[0],-1)
                #model.load_state_dict(state_dict)                
                
                log_probs = model(images)

                #log_probs[:,client_class_idx] = 0.

                if self.args.loss=='ce':
                    loss = self.criterion(log_probs, labels)
                else:
                    loss = self.criterion(log_probs, labels,indicate_class)
 
                #FEDPROX
                #mu =0.01
                #fed_prox_reg = 0.0
                #for param, global_param in zip(model.parameters(), global_model.parameters()):
                 #   fed_prox_reg += ((mu / 2) * torch.norm((param - global_param))**2)
                #loss += fed_prox_reg
                    
                #reg= torch.norm((model.classifier.weight -torch.mean(model.classifier.weight,dim=1)),p=1)
                #loss += reg
                '''with torch.no_grad():
                    global_model.eval()
                    output_teacher = global_model(images)
                if global_round > 0:
                    loss += loss_fn_kd(log_probs, labels, output_teacher,alpha=1.0,temperature=2.0)'''
          
                ''' original loss!!'''
                loss.backward()
    
          
                # gradient 확인용 - how does BN
                #conv_grad.append(model.conv1.weight.grad.clone().to('cpu'))
                if self.args.optimizer != 'clip':
                    total_norm.append(check_norm(model))
                """
                if self.args.model == 'cnn' or self.args.model == 'cnn_ws':
                    fc_grad.append(model.fc3.weight.grad.clone().detach().to('cpu'))
                else:
                    fc_grad.append(model.linear.weight.grad.clone().detach().to('cpu'))
                """
                if self.args.optimizer == 'sam':
                    optimizer.first_step(zero_grad=True)
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                elif self.args.optimizer == 'clip':
                    max_norm = 0.25
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    #total_norm.append(check_norm(model))
                    optimizer.step()
                else:  # sam이 아닌 경우
                    optimizer.step()
                if self.args.local_decay:
                    scheduler.step()
                # print(optimizer.param_groups[0]['lr']) # - lr decay 체크용
                if self.args.verbose:
                    print('|Client : {} Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        idx_user, global_round + 1, iter_ + 1, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                  100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                # itr loss 확인용 - how does BN
                loss_list.append(loss.item())
                
                if self.args.verbose == 0:
                    del images
                    del labels
                    torch.cuda.empty_cache()
            #print(total_norm) # gradient 확인용
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.model = copy.deepcopy(model)

           #print('epoch = %04d, loss = %.4f' % (ep, loss_avg/self.args.local_bs))
#            ep+=1

        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self):
        """ Returns the inference accuracy and loss.
        """

        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        label_entropy =0.0
        idx=0
        #with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(self.trainloader):

            self.model.zero_grad()

            images, labels = images.to(self.device), labels.to(self.device)
            # Inference
            outputs = self.model(images)
            #outputs = self.global_model(images)
            batch_loss = self.criterion(outputs, labels)

            loss += batch_loss.item()
            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            pred = F.softmax(outputs,dim=1)
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            entropy = (-1*torch.sum(pred * torch.log(pred), dim=1))
            label_entropy += entropy.sum()
            idx+=1

        accuracy = correct / total
        return accuracy, loss,label_entropy/idx
    
    def personal_infernece(self,class_dict,test_dataset):
        """ Returns the inference accuracy and loss.
        """

        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        label_entropy =0.0
        idx=0
        class_dict = list(class_dict)
        if self.args.dataset == 'tiny_imagenet' or self.args.dataset == 'imagenet':
            batch_size = 50
        else:
            batch_size = 100
        testloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False,num_workers=4,pin_memory=True)
        class_acc = [0 for i in range(self.args.num_classes)]

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                
                if self.args.dataset == "imagenet" and batch_idx not in class_dict:
                    continue
                self.model.zero_grad()

                images, labels = images.to(self.device), labels.to(self.device)
                # Inference
                outputs = self.model(images)
                #outputs = self.global_model(images)
                batch_loss = self.criterion(outputs, labels)

                loss += batch_loss.item()
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                class_correct  = torch.sum(torch.eq(pred_labels, labels)).item()
                correct += class_correct
                total += len(labels)
                class_acc[batch_idx] = class_correct
        accuracy = correct / total
        class_acc = np.array(class_acc)[class_dict]
        class_acc = np.sum(class_acc) / (batch_size * len(class_dict))
        return accuracy,class_acc
    
    def global_inference(self):
        """ Returns the inference accuracy and loss.
        """

        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        label_entropy =0.0
        idx=0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                self.model.zero_grad()

                images, labels = images.to(self.device), labels.to(self.device)
                # Inference
                outputs = self.global_model(images)
                #outputs = self.global_model(images)
                batch_loss = self.criterion(outputs, labels)

                loss += batch_loss.item()
                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                #label_one_hot = torch.nn.functional.one_hot(labels, 10).float().to(self.device)
                #label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
                pred = F.softmax(outputs,dim=1)
                entropy = (-1*torch.sum(pred * torch.log(pred), dim=1))
                label_entropy += entropy.sum()
                idx+=1

        accuracy = correct / total
        return label_entropy/idx, loss


def test_inference(args, test_model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """
    model = test_model
   # model.to(device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    if args.dataset == 'tiny_imagenet' or args.dataset == 'imagenet':
        data, labels = test_dataset.imgs, test_dataset.targets
        batch_size = 50
    else:
        data, labels = test_dataset.data, test_dataset.targets
        sort_index = np.argsort(labels)
        data = data[sort_index]
        labels = np.array(labels)
        labels = labels[sort_index]
        
        test_dataset.data = data
        test_dataset.targets = labels
        batch_size = 100

    testloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False,num_workers=4,pin_memory=True)
    # cifar 10 기준 50번(100*50)이 한번의 class
    if args.dataset =='cifar10':
        class_acc = [0 for i in range(10)]
    elif args.dataset == 'tiny_imagenet':
        class_acc = [0 for i in range(200)]
    elif args.dataset == 'imagenet':
        class_acc = [0 for i in range(1000)]
    else:
        class_acc = [0 for i in range(100)]

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
            #if args.dataset =='cifar10':
            #    class_acc[(batch_idx)//10] += class_correct
            #else:
            #    class_acc[batch_idx] = class_correct        
  

        
    #for i in range(10): class_acc[i]/=1000 
    accuracy = correct / len(test_dataset)
    del images
    del labels
    #print(class_acc)
    return accuracy ,loss/len(iter(testloader)),class_acc 

def test_inference_tsne(args, test_model, test_dataset, device):
    """ Returns the test accuracy and loss.
    """
    model = copy.deepcopy(test_model)

   # model.to(device)
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    data, labels = test_dataset.data, test_dataset.targets
    sort_index = np.argsort(labels)
    data = data[sort_index]
    labels = np.array(labels)
    labels = labels[sort_index]
    
    test_dataset.data = data
    test_dataset.targets = labels
    testloader = DataLoader(test_dataset, batch_size=100,
                            shuffle=False)
    # cifar 10 기준 50번(100*50)이 한번의 class
    class_acc = [0 for i in range(10)]
    
    for module in model.modules():
        if module.__class__.__name__ == 'Sequential':
            features = tsnehook(module)
    
    #embeddings = np.zeros(shape=(100,128,4,4))
    test_predictions = []
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
            class_acc[(batch_idx)//10] += class_correct
            # if pred_labels -> labels로 바꾸면 G.T tsne확인가능
            if args.tsne_pred ==1:
                test_predictions.extend(pred_labels.detach().cpu().tolist())
            else:
                test_predictions.extend(labels.detach().cpu().tolist())

            if batch_idx==0:
                embeddings = features.output.detach().cpu().numpy()
            else:
                embeddings = np.concatenate([embeddings,features.output.detach().cpu().numpy()],axis=0)
        

        test_embeddings = embeddings.reshape(embeddings.shape[0],-1)
        test_predictions = np.array(test_predictions)

        tsne_proj =  TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(test_embeddings)
        # Plot those points as a scatter plot and label them based on the pred labels
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(16,10))


        num_categories = 10
        for lab in range(num_categories):
            indices = test_predictions==lab
            ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.show()
        
    del features
    #for i in range(10): class_acc[i]/=1000 
    accuracy = correct / len(test_dataset)
    del images
    del labels
    #print(class_acc)
    return accuracy ,loss/len(iter(testloader)),class_acc

def pnorm(weights, p):
    normB = torch.norm(weights, 2, 1)
    ws = weights.clone()
    for i in range(weights.size(0)):
        ws[i] = ws[i] / torch.pow(normB[i], p)
    return ws

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu



def match_loss(gw_syn, gw_real, device):
    dis = torch.tensor(0.0).to(device)

    #if args.dis_metric == 'ours':
    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(gwr, gws)
    """
    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)
    
    else:
        exit('DC error: unknown distance function')
    """
    return dis



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return 0
    
    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def epoch(mode, dataloader, net, optimizer, criterion, param_augment, device):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(device)
        #if mode == 'train' and param_augment != None:
        #    img = augment(img, param_augment, device=device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


    

def evaluate_synset(it_eval, net, images_train, labels_train, test_dataset, learningrate, batchsize_train, param_augment, device, Epoch = 100):
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)



    lr = float(learningrate)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    dst_train = torch.utils.data.TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batchsize_train, shuffle=True, num_workers=0)

    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, param_augment, device)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            #optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)

    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, param_augment, device)
    print('Evaluate_%02d: epoch = %04d  train loss = %.6f train acc = %.4f, test acc = %.4f' % ( it_eval, Epoch, loss_train, acc_train, acc_test))

    return net, acc_train, acc_test

def interpolation_model(model, global_model, ratio=0.5):
    """
    Returns the average of the weights.
    """
    m = model.state_dict()
    g = global_model.state_dict()
    for key in g.keys():
        g[key] =g[key]*ratio + m[key] * (1 - ratio)
    model.load_state_dict(g)
    return model


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



def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = alpha
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


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


        self.output = output
        # must have no output

    def close(self):
        self.hook.remove()


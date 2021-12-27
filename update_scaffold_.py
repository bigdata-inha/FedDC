#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import torch
from torch import nn
from torch.nn import parameter
from torchvision import transforms
from sam import SAM
from sgd_agc import SGD_AGC
from agc import AGC
from torch.utils.data import DataLoader, Dataset
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
import math
from FedNova import FedProxOptimizer, FedProx,SCAFFOLDOptimizer


class LocalUpdate(object):
    def __init__(self, args, logger, train_loader, device, syn_client):
        # idxs= client의 data index ex) 5000개 data의 index집합

        self.args = args
        self.logger = logger
        self.trainloader = train_loader

        self.device = device
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss()

        ipc = 1
        num_classes = 10
        channel = 3
        im_size = (32, 32)
        ''' initialize the synthetic data '''


    def update_weights(self, model, global_round, idx_user,syn_dataset,images_all,train_dataset,test_dataset, c_global, c_local):
        # Set mode to train model
#        model.to(self.device)
#        model.train()
        epoch_loss = []
        total_norm = []
        loss_list = []
        conv_grad = []
        fc_grad = []
        global_model = copy.deepcopy(model)
        self.global_model = global_model
        global_model_para = copy.deepcopy(global_model.state_dict())

        # 시드 고정
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.args.seed+ idx_user)
        torch.cuda.manual_seed(self.args.seed+ idx_user)
        random.seed(self.args.seed+ idx_user)
        np.random.seed(self.args.seed+ idx_user)

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=1e-3)
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
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9
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
        elif self.args.optimizer == 'scaffold':
            optimizer = SCAFFOLDOptimizer( model.parameters(), lr=self.args.lr,weight_decay=1e-3)
  
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4
                                    )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
        
        origin_dataset = self.trainloader.dataset     

        ###################################################################################
        # data 생성 과정
        idxs = origin_dataset.idxs
        ipc = 1
        num_classes = 10
        channel = 3
        im_size = (32, 32)
        ''' organize the real dataset '''
        train_x = [torch.unsqueeze(train_dataset[i][0], dim=0) for i in idxs]
        train_x = torch.cat(train_x, dim=0).to(self.device)
        images_all = images_all[idxs]
        labels_all = origin_dataset.dataset.targets
        indices_class = [[] for c in range(num_classes)]

        dst_train =train_dataset
        #labels_all = [dst_train[i][1] for i in range(len(dst_train))]


        #for i, lab in enumerate(labels_all):
        #    indices_class[lab].append(i)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)
        labels_all = labels_all[idxs]
        for i, lab in enumerate(labels_all):
           indices_class[lab].append(i)

        """
        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        """
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        ###################################################################################
        
        param_augment = dict()
        param_augment['strategy'] = 'crop_gray_brightness_contrast_saturation_hue'
        
        if global_round!=0:
            syn_idx = len(syn_dataset)
            syn_trainloader = DataLoader(syn_dataset,batch_size=syn_idx,shuffle=True)

        #syn_it = self.args.local_bs / syn_idx
               
        local_dataset = torch.utils.data.TensorDataset(train_x, labels_all)
        #if global_round !=0:
        #  self.trainloader = DataLoader(local_dataset,
         #                        batch_size=syn_idx,shuffle=True)
        ratio_list = [i for i in range(0,10)]

        #if global_round !=0:
        #       self.trainloader = DataLoader(torch.utils.data.ConcatDataset([local_dataset, syn_dataset]),
        #                           batch_size=self.args.local_bs,shuffle=True)

    
        c_global_para = c_global.state_dict()
        c_local_para = c_local
        cnt = 0

        global_param = list(global_model.parameters())
        for iter_ in range(self.args.local_ep):
            batch_loss = []
            loss_avg = 0
           
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                '''
                if global_round!=0:
                    syn_img, syn_lab = next(iter(syn_trainloader))
                    syn_img = syn_img[torch.arange(images.shape[0]),:]
                    syn_lab = syn_lab[torch.arange(labels.shape[0])]
                    syn_img.requires_grad=True
                    syn_lab.requires_grad=False
                '''
                #torch.manual_seed(self.args.seed+ idx_user+batch_idx)
                #images = augment(images, param_augment)

                #                 #if batch_idx==0:
                #    print('idx_user: {}, label: {}'.format(idx_user, labels[0]))
                images.requires_grad=True
                labels.requires_grad=False
                
                '''
                net_parameters = list(global_model.parameters())
                output_syn = global_model(images)
                loss_syn = self.criterion(output_syn, labels)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters)
                gw_syn = list((_.detach().clone() for _ in gw_syn))
                '''
                '''
                #mix img
                if global_round!=0:
                    images= images+syn_img
                optimizer.zero_grad()
                
                log_probs = model(images)

                if global_round!=0:                  
                    loss = self.criterion(log_probs, labels) *0.8 + self.criterion(log_probs,syn_lab) *0.2
                else:
                    loss = self.criterion(log_probs, labels)
                '''
                optimizer.zero_grad()

                log_probs = model(images)
                output = log_probs 
                loss = self.criterion(output, labels)
                '''
                ratio = ratio_list[batch_idx%10]
                ratio*=0.1
                #ratio = 0
                model = interpolation_model(model,global_model,ratio)
                output_teacher = model(images)
                loss = self.criterion(output_teacher, labels)
                '''
                #cos_loss = match_loss(global_param,list(model.parameters()),self.device)
                #print("loss:{},cos_loss:{}".format(loss,cos_loss))
                #loss += cos_loss *0.1
                #FEDPROX
                #mu =0.01
                #fed_prox_reg = 0.0
                #for param, global_param in zip(model.parameters(), global_model.parameters()):
                 #   fed_prox_reg += ((mu / 2) * torch.norm((param - global_param))**2)
                #loss += fed_prox_reg
                
                ''' update synthetic data '''
 
                #with torch.no_grad():
                #    output_teacher = global_model(images)

                #if global_round>1:
                #    loss = loss_fn_kd(log_probs, labels, output_teacher,alpha=0.8,temperature=2.0)
      
                #net = copy.deepcopy(model)
                #net_parameters = list(net.parameters())
                #gw_real_ts = torch.autograd.grad(loss, net_parameters)
                #model_parameters = list(model.parameters())
                #gw_real = torch.autograd.grad(loss, model_parameters, create_graph=True)
                
                #loss_dc = 0.001 * match_loss(gw_syn, gw_real, self.device)
                #loss = loss_dc+loss
                ''' original loss!!'''
                loss.backward()
    
                '''
                img_syn = self.image_syn

                output_syn = net(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                loss_dc += match_loss(gw_syn, gw_real, self.device)

                optimizer_img.zero_grad()
                
                loss_dc.backward()
                #max_norm = 0.25
                #torch.nn.utils.clip_grad_norm_([self.image_syn,], max_norm)
                optimizer_img.step()
                loss_avg += loss_dc.item()        
                '''
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
                elif self.args.optimizer == 'scaffold':
                    #if global_round==0:
                   #     optimizer.step()
                    #else:
                    optimizer.step(server_controls, client_control)


                else:  # sam이 아닌 경우
                    optimizer.step()
                if self.args.local_decay:
                    scheduler.step()
               
         
                # print(optimizer.param_groups[0]['lr']) # - lr decay 체크용
                model_param = model.state_dict()
                # SCAFFOLD
                for key in model_param:
                    model_param[key] = model_param[key] - self.args.lr * (c_global_para[key] - c_local_para[key])
                model.load_state_dict(model_param)
         
        
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
              
                cnt += 1

                if self.args.verbose == 0:
                    del images
                    del labels
                    torch.cuda.empty_cache()
            #print(total_norm) # gradient 확인용
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            self.model = copy.deepcopy(model)
 
        
        # For SCAFFOLD
        c_new_para = c_local
        c_delta_para = copy.deepcopy(c_local)
        #위에서 구현 -> global_model_para = global_model.state_dict()
        for key in model_param:            
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - model_param[key]) / (cnt * self.args.lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]
        c_local = copy.deepcopy(c_new_para)
        self.model = model


        dst_syn_train =0
        
        ''' training '''

       
        client_class = []
        for c, ind in enumerate(indices_class):
           if len(ind)>0: 
            client_class.append(c)
        lab_syn = torch.tensor([np.ones(ipc)*i for i in client_class], dtype=torch.long, requires_grad=False, device=self.device).view(-1)
        self.image_syn = torch.randn(size=(len(client_class)*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)
  
        optimizer_img = torch.optim.SGD([self.image_syn, ], lr=0.5, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self.device)
 
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        Iteration = -1
        outer_loop, inner_loop = 3,1
        batch_real = 50
        batch_train = 50
        eval_it_pool = [100,200,500,800,1000]
        epoch_eval_train = 100
        param_augment = dict()
        param_augment['strategy'] = 'crop_gray_brightness_contrast_saturation_hue'
        
        # starting real
        for idx,c in enumerate(client_class):
            self.image_syn.data[idx*ipc:(idx+1)*ipc] = get_images(c, ipc).detach().data
        
        ratio_list = [i for i in range(0,100)]
        np.random.shuffle(ratio_list) 
        for it in range(Iteration+1):
            #torch.manual_seed(self.args.seed+ idx_user+it)
            if it in eval_it_pool:
                accs = []

                net_eval = copy.deepcopy(self.global_model) # get a random model
                #img_real = get_images_all(10)
                image_syn_eval, label_syn_eval = copy.deepcopy(self.image_syn.detach()), copy.deepcopy(lab_syn.detach()) # avoid any unaware modification
                _, acc_train, acc_test = evaluate_synset(it, net_eval, image_syn_eval, label_syn_eval, test_dataset, 0.01, batch_train,None, self.device, epoch_eval_train)
                accs.append(acc_test)
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), 'ConvNet', np.mean(accs), np.std(accs)))

                save_name = os.path.join('result_visual', 'client_%d_vis_%s_%s_%dipc_iter%d.png'%(idx_user, self.args.dataset, self.args.model, 1, it))
                image_syn_vis = copy.deepcopy(self.image_syn.detach().cpu())
                #image_syn_vis = augment(image_syn_vis,param_augment)
                _,channel,_,_ = image_syn_vis.shape
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=1) # Trying normalize = True/False may get better visual effects.
                # The generated images would be slightly different from the visualization results in the paper, because of the initialization and normalization of pixels.
            
            ''' Train synthetic data '''
            net = copy.deepcopy(model) # get a random model
            ratio = ratio_list[it%100]
            ratio*=0.01
            #ratio = 0
            net = interpolation_model(net,self.global_model,ratio)
            #net = ConvNet(3,10,128, 3, 'relu', 'instancenorm', 'avgpooling',(32,32)).to(self.device)
            net.train()
            #normal_dist = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))
            #with torch.no_grad():
            #    for param in net.parameters():
            #        t = normal_dist.sample((param.view(-1).size())).reshape(param.size()).to(self.device)
            #        param.add_(t)
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            
            #et_syn = copy.deepcopy(net)
            #net_syn_parameters = list(net_syn.parameters())
            #optimizer_net_syn = torch.optim.SGD(net_syn.parameters(), lr=0.01, momentum=0.5)  # optimizer_img for synthetic data
            #optimizer_net_syn.zero_grad()           
            
            for ol in range(outer_loop):
                
                optimizer_net.zero_grad()

                all_dataset = torch.utils.data.TensorDataset(images_all, labels_all)
                dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=batch_train, shuffle=True, num_workers=0)


                
                for i_batch, datum in enumerate(dataloader):
                    img = datum[0].float().to(self.device)
                    
                    lab = datum[1].long().to(self.device)
                    optimizer_net.zero_grad()

                    output = net(img)
                    loss_real = criterion(output, lab)
                    #loss_real.backward(retain_graph=True)
                    #gw_real = [param.grad for param in net.parameters()] 
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    
                    #optimizer_net.step()
                    
                    output_syn = net(self.image_syn)

                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    loss = match_loss(gw_syn, gw_real, self.device)
                    
                    optimizer_img.zero_grad()
                    loss.backward()
                    optimizer_img.step()
                    loss_avg += loss.item()
                    #optimizer_net.step()
                loss_avg /= (outer_loop)
                
                if ol == outer_loop - 1:
                    break
                
                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(self.image_syn.detach()), copy.deepcopy(lab_syn.detach())  # avoid any unaware modification
                dst_syn_train = torch.utils.data.TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=batch_train, shuffle=True, num_workers=0)
                
                for il in range(inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, None, self.device)

            #if it%10 == 0:
            #   print('iter = %04d, loss = %.4f' % ( it, loss_avg))
                    
        with torch.no_grad():
            output_client = model(self.image_syn)
        '''
        class_image_syn_1, class_image_syn_2 = torch.sum(self.image_syn[:ipc],dim=0).reshape(1,3,32,32),torch.sum(self.image_syn[ipc:ipc*2],dim=0).reshape(1,3,32,32)
        class_label_syn = lab_syn = torch.tensor([np.ones(1)*i for i in client_class], dtype=torch.long, requires_grad=False, device=self.device).view(-1)
        class_image_syn = torch.cat((class_image_syn_1,class_image_syn_2),dim=0)
        class_image_syn = torch.div(class_image_syn,ipc)
        
        save_name = os.path.join('result_visual', 'result_client_%d_vis_%s_%s_%dipc_iter%d.png'%(idx_user, self.args.dataset, self.args.model, 1, it))
        image_syn_vis = copy.deepcopy(class_image_syn.detach().cpu())
        for ch in range(channel):
            image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
        image_syn_vis[image_syn_vis<0] = 0.0
        image_syn_vis[image_syn_vis>1] = 1.0
        save_image(image_syn_vis, save_name, nrow=1) # Trying normalize = True/False may get better visual effects. 
        '''
        class_image_syn, class_label_syn = self.image_syn, lab_syn
        #class_image_syn = augment_batch(class_image_syn,param_augment)
        if not client_class:
            dst_syn_train = None
        else:
            dst_syn_train = torch.utils.data.TensorDataset(class_image_syn.detach(),class_label_syn.detach())

        indicate_class = []
        for i in indices_class:
            indicate_class.append(len(i))

        #model.classifier.weight.data = pnorm(model.classifier.weight.data,0.5)
        
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), loss_list, conv_grad, fc_grad, total_norm, dst_syn_train, output_client,indicate_class, c_delta_para, c_local

    def inference(self):
        """ Returns the inference accuracy and loss.
        """

        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
   
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

        accuracy = correct / total
        return accuracy, loss
    

def test_inference(args, test_model, test_dataset, device):
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


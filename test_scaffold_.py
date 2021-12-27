#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import copy
from threading import local
import time
import pickle
import numpy as np
import easydict
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch import nn

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update_scaffold_ import LocalUpdate, test_inference,test_inference_tsne
from networks import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar_fedVC, CNNCifar_WS, AlexNet, ConvNet
from utils import aggregate, average_weights_class_normalize, get_dataset, average_weights, eval_smooth, average, exp_details,augment, get_logger, CSVLogger, set_model_global_grads, average_weights_uniform, exp_decay_schedule_builder
from sampling import client_choice
from fed_cifar100 import load_partition_data_federated_cifar100
from resnet_gn import resnet18
from resnet import ResNet32_test, ResNet32_nobn, ResNet50
from pre_resnet import PreActResNet18, PreActResNet18_nobn, PreActResNet50
from vgg import vgg11_bn, vgg11, vgg11_cos,vgg11_bn_cos
from FedNova import *
from sam import SAM
import random
import logging
import datetime
import torchsummary
from optrepo import OptRepo
from pcgrad import pc_grad_update
# In[6]:


def main_test(args):
    start_time = time.time()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    # define paths

    logger = SummaryWriter('../logs')

    # easydict 사용하는 경우 주석처리
    # args = args_parser()

    # checkpoint 생성위치
    args.save_path = os.path.join(args.save_path, args.exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    save_path_tmp = os.path.join(args.save_path, 'tmp_{}'.format(now))
    if not os.path.exists(save_path_tmp):
        os.makedirs(save_path_tmp)
    SAVE_PATH = os.path.join(args.save_path, '{}_{}_{}_T[{}]_C[{}]_iid[{}]_E[{}]_B[{}]'.
                             format(args.dataset, args.model,args.norm, args.epochs, args.frac, args.iid,
                                    args.local_ep, args.local_bs))

    # 시드 고정
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


    if not os.path.exists('result_visual'):
        os.mkdir('result_visual')

#    torch.cuda.set_device(0)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device('cpu')
    # log 파일 생성
    log_path = os.path.join('../logs', args.exp_folder)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    loggertxt = get_logger(
        os.path.join(log_path, '{}_{}_{}_{}.log'.format(args.model, args.optimizer, args.norm, now)))
    logging.info(args)
    # csv
    csv_save = '../csv/' + args.exp_folder +'/'+ now 
    csv_path = os.path.join(csv_save, 'accuracy.csv')
    csv_logger_keys = ['train_loss', 'accuracy']
    csvlogger = CSVLogger(csv_path, csv_logger_keys)

    # load dataset and user groups
    train_dataset, test_dataset, client_loader_dict, real_dataset, client_class_set = get_dataset(args)

    # cifar-100의 경우 자동 설정
    if args.dataset == 'cifar100' or args.dataset == 'fedcifar100':
        args.num_classes = 100
    elif args.dataset == 'cifar10':
        args.num_classes = 10

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural network
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar10':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'cifar100':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    elif args.model == 'cnn_vc':
        global_model = CNNCifar_fedVC(args=args,weight_stand=0)
    elif args.model == 'cnn_vc_ws':
        global_model = CNNCifar_fedVC(args=args,weight_stand=1)

    elif args.model == 'resnet18_ws'and args.norm=='nothing':
        global_model = PreActResNet18_nobn(num_classes=args.num_classes, weight_stand=1, group_norm=args.norm)
    elif args.model == 'resnet18'and args.norm=='nothing':
        global_model = PreActResNet18_nobn(num_classes=args.num_classes, weight_stand=0, group_norm=args.norm)
    elif args.model == 'resnet18_wc'and args.norm=='nothing':
        global_model = PreActResNet18_nobn(num_classes=args.num_classes, weight_stand=2, group_norm=args.norm)

    elif args.model == 'resnet18_ws':
        global_model = PreActResNet18(num_classes=args.num_classes, weight_stand=1, group_norm=args.norm)
    elif args.model == 'resnet18':
        global_model = PreActResNet18(num_classes=args.num_classes, weight_stand=0, group_norm=args.norm)
    elif args.model == 'resnet18_wc':
        global_model = PreActResNet18(num_classes=args.num_classes, weight_stand=2, group_norm=args.norm)

    elif args.model == 'resnet32_ws' and args.norm=='nothing':
        global_model = ResNet32_nobn(num_classes=args.num_classes, weight_stand=1, group_norm=args.norm)
    elif args.model == 'resnet32_wc ' and args.norm=='nothing':
        global_model = ResNet32_nobn(num_classes=args.num_classes, weight_stand=2, group_norm=args.norm)
    elif args.model == 'resnet32'and args.norm=='nothing':
        global_model = ResNet32_nobn(num_classes=args.num_classes, weight_stand=0, group_norm=args.norm)
    
    elif args.model == 'resnet32_ws':
        global_model = ResNet32_test(num_classes=args.num_classes, weight_stand=1, group_norm=args.norm)
    elif args.model == 'resnet32':
        global_model = ResNet32_test(num_classes=args.num_classes, weight_stand=0, group_norm=args.norm)
    elif args.model == 'resnet32_wc':
        global_model = ResNet32_test(num_classes=args.num_classes, weight_stand=2, group_norm=args.norm)

    elif args.model == 'vgg':
        global_model = vgg11()
    elif args.model == 'vgg_bn':
        global_model = vgg11_bn()
    elif args.model == 'vgg_cos':
        global_model = vgg11_cos()
    elif args.model == 'vgg_bn_cos':
        global_model = vgg11_bn_cos()
    elif args.model == 'cnn_ws':
        global_model = CNNCifar_WS(args=args)
    
    elif args.model == 'alexnet':
        global_model = AlexNet()
    elif args.model == 'ConvNet':
        if args.cosine_norm == 0:
            global_model = ConvNet(3,10,128, 3, 'relu', args.norm, 'maxpooling',(32,32))
        else:
            global_model = ConvNet(3,10,128, 3, 'relu', args.norm, 'cos_norm',(32,32))
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    loggertxt.info(global_model)    
    if args.dataset == 'fedcifar100':
        loggertxt.info(torchsummary.summary(copy.deepcopy(global_model),(3,24,24),device='cpu'))
    else:
        loggertxt.info(torchsummary.summary(copy.deepcopy(global_model),(3,32,32),device='cpu'))

    # fedBN처럼 gn no communication 용
    #client_models = [copy.deepcopy(global_model) for idx in range(args.num_users)]

    # copy weights
    global_weights = global_model.state_dict()
    #for i, (name, layer) in enumerate(global_weights.items()):
    #    print(i, ' ', name, ' ', layer.size())
    global_model.to(device)
    old_model = copy.deepcopy(global_model)

    best_accuracy =0.0
    best_round = 0
    # Training
    train_loss  = []
    val_acc_list, test_loss_list,test_class_acc_list = [], [], []


    # how does help BN 확인용
    client_loss = [[] for i in range(args.num_users)]
    client_conv_grad = [[] for i in range(args.num_users)]
    client_fc_grad = [[] for i in range(args.num_users)]
    client_total_grad_norm = [[] for i in range(args.num_users)]
    # 전체 loss 추적용 -how does help BN
    # 재시작
    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        global_model.load_state_dict(checkpoint['global_model'])
        #if args.hold_normalize:
        #    for client_idx in range(args.num_users):
        #        client_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        #else:
        #    for client_idx in range(args.num_users):
        #        client_models[client_idx].load_state_dict(checkpoint['global_model'])
        #resume_iter = int(checkpoint['a_iter']) + 1
        resume_iter = 100
        print('!! Resume trainig form epoch {} !! '.format(resume_iter))
    else:
        resume_iter = 0

    if args.server_opt == 'adam':
        opt = OptRepo.name2cls(args.server_opt)(global_model.parameters(), betas=(0.9, 0.99) ,lr=args.server_lr, eps=1e-3)
    elif args.server_opt == 'sgdm':
        opt = OptRepo.name2cls('sgd')(global_model.parameters(), lr=args.server_lr, momentum=0.9)
        server_momntum = copy.deepcopy(global_model.state_dict())
        for key in server_momntum:
            server_momntum[key] = 0
    elif args.server_opt == 'sam':
        base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
        opt = SAM(global_model.parameters(), base_optimizer, lr=args.server_lr,  weight_decay=1e-3)
    else:
        opt = OptRepo.name2cls(args.server_opt)(global_model.parameters(), lr=args.server_lr)
   
    # global_grads = list(global_model.parameters())
    syn_client = [[] for i in range(args.num_users)]
    syn_total_dataset = []
    syn_round_dataset = None

    images_all = []
    images_all = [torch.unsqueeze(real_dataset[i][0], dim=0) for i in range(len(real_dataset))]
    images_all = torch.cat(images_all, dim=0).to(device)
    # learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.1,step_size=500)
    #server_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1)
    client_chocie = client_choice(args, args.num_users)

    
    c_global = copy.deepcopy(global_model)
    c_global_param = c_global.state_dict()
    for key in c_global_param:
        c_global_param[key] = torch.zeros_like(c_global_param[key])
    c_global.load_state_dict(c_global_param)
    
    client_models = [copy.deepcopy(c_global_param) for idx in range(args.num_users)]

    print([len(client_loader_dict[r].dataset) for r in range(args.num_users)])

    # start training
    accumlate_indicate_class = []
    round_indicate_class = []
    for epoch in tqdm(range(resume_iter, args.epochs)):

        # DD용
        syn_dataset = []
        output_clients = []

        client_indicate_class = []
        local_weights, local_losses = [], []
        train_accuracy = 0.0
        if args.verbose:
            print(f'\n | Global Training Round : {epoch + 1} |\n')

        total_delta = copy.deepcopy(global_model.state_dict())
        for key in total_delta:
            total_delta[key] = 0.0
     
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
      
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #idxs_users = [i for i in range(10)]
        idxs_users = client_chocie.client_sampling(range(args.num_users), m, args.seed, epoch)
        print([len(client_loader_dict[r].dataset) for r in idxs_users])
        #print(idxs_users)
        for it, idx in enumerate(idxs_users):
            """
            for key in global_model.state_dict().keys():
                if args.hold_normalize:
                    if 'bn' not in key:
                        client_models[idx].state_dict()[key].data.copy_(global_model.state_dict()[key])
                else:
                    client_models[idx].state_dict()[key].data.copy_(global_model.state_dict()[key])
            """    
           

            torch.cuda.empty_cache()
            #client_models[idx].load_state_dict(global_model.state_dict())
            #client_models[idx].train()
            global_model.train()

            local_model = LocalUpdate(args=args, logger=logger, train_loader=client_loader_dict[idx], device=device, syn_client = syn_client[idx])
            w, loss, batch_loss, conv_grad, fc_grad, total_gard_norm, syn_train,output_client,indices_class,c_delta_para, c_local = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, idx_user=idx, syn_dataset = syn_round_dataset, images_all=images_all,
                                    train_dataset=train_dataset,test_dataset=test_dataset, c_global=c_global, c_local=client_models[idx])
            for key in total_delta:
                total_delta[key] += c_delta_para[key]

            #with torch.no_grad():
            #client_train_accuracy,_ = local_model.inference()

            #train_accuracy += client_train_accuracy
            local_weights.append(copy.deepcopy(w))
            # client의 1 epoch에서의 평균 loss값  ex)0.35(즉, batch loss들의 평균)
            local_losses.append(copy.deepcopy(loss))

            syn_dataset.append(syn_train)
            syn_total_dataset.append(syn_train)
            syn_client[idx] = copy.deepcopy(syn_train)
            output_clients.append(output_client)
            client_indicate_class.append(indices_class)

            # 전체 round scheduler
            #  scheduler.step()
        
            # loss graph용 -> client당 loss값 진행 저장 -> 모두 client별로 저장.
            client_loss[idx].append(batch_loss)
            #if idx==0:
            #    client_conv_grad[idx].append(conv_grad)
            #    client_fc_grad[idx].append(fc_grad)
            client_total_grad_norm[idx].append(total_gard_norm)
            # print(total_gard_norm)
            # gn, bn 복사
            client_models[idx] = copy.deepcopy(c_local)
            del local_model
            del w
        
        for key in total_delta:
            total_delta[key] /= len(idxs_users)
        c_global_para = c_global.state_dict()
        for key in c_global_para:
           # c_global_para[key] = c_global_para[key].type(torch.cuda.DoubleTensor)
            c_global_para[key] += total_delta[key].type(torch.cuda.DoubleTensor)
        c_global.load_state_dict(c_global_para)
    

        syn_round_dataset = copy.deepcopy(torch.utils.data.ConcatDataset(syn_dataset))
        #print(len(syn_round_dataset))
        #old_model = copy.deepcopy(global_model)
        #old_weights = copy.deepcopy(global_model.state_dict())
        
        #local_weights = update_locals(global_model,local_weights, syn_round_dataset,output_clients, device,test_dataset)
        # update global weights
        
        #global_model = aggregate_scaffold(global_model,server_controls,total_delta_controls=total_delta,total_delta_model=total_delta_model,num_users=args.num_users,idxs_users=idxs_users)
            
                    
        client_indicate_class = torch.tensor(client_indicate_class)
        class_total_data_points = torch.sum(client_indicate_class.T,dim=1)
        if epoch == 0:
            accumlate_indicate_class.append(class_total_data_points)
        else:
            accumlate_indicate_class.append(accumlate_indicate_class[epoch-1]+class_total_data_points)
        round_indicate_class.append(class_total_data_points)

        global_weights = average_weights(local_weights, client_loader_dict, idxs_users)
        
       # global_weights = average_weights_class_normalize(local_weights,client_loader_dict,idxs_users,torch.tensor(client_indicate_class).to(device))

        """        
        for key in global_weights.keys():
            global_weights[key] = global_weights[key] * 0.5
            global_weights[key] += old_weights[key] * 0.5
        """

        # 원래 위치
        # server opt.를 사용할 경우 주석
        #global_model.load_state_dict(global_weights)
        
        # set server optim
        opt.zero_grad()
        opt_state = opt.state_dict()
        # 위 avg와 동시사용 불가. load state dict
        global_model = set_model_global_grads(global_weights, global_model)
        
        """        
        if epoch > 1:
            for param1, param2 in zip(global_model.parameters(), old_model.parameters()):
                param1.data *= 0.5
                param1.data += param2.data * (1.0 - 0.5)
        """
        
        # server optimization
        if args.server_opt == 'adam':
            opt = OptRepo.name2cls(args.server_opt)(global_model.parameters(), betas=(0.9, 0.999) ,lr=args.server_lr, eps=1e-1)
            global_model.load_state_dict(global_weights)
        elif args.server_opt == 'sgdm':
            opt = OptRepo.name2cls('sgd')(global_model.parameters(), lr=args.server_lr, momentum=0.9)
            #delta_w = copy.deepcopy(global_weights)
            #for key in delta_w:
            #    delta_w[key]= old_weights[key] - global_weights[key]
            #    server_momntum[key] = 0.9 * server_momntum[key] + (1 - 0.9) * delta_w[key]
            #    global_weights[key] = old_weights[key] - server_momntum[key]
            
            #global_model.load_state_dict(global_weights)
        elif args.server_opt == 'sam':
            opt = SAM(global_model.parameters(), base_optimizer, lr=args.server_lr,  weight_decay=1e-3)
        else:
            opt = OptRepo.name2cls(args.server_opt)(global_model.parameters(), lr=args.server_lr)

        opt.load_state_dict(opt_state)
        if args.server_opt != 'sam':
            opt.step()
        else:
            opt.first_step(zero_grad=True)
            global_model = set_model_global_grads(global_weights, global_model)
            opt.second_step(zero_grad=True)

        if args.server_epoch > 0:
            global_model.eval()
            test_accuracy, test_loss, _ = test_inference(args, global_model, test_dataset, device=device)
            loggertxt.info('Test Accuracy: {:.2f}% Before Training'.format(100 * test_accuracy))
            loggertxt.info(f'Test Loss: {test_loss} Before Training\n')

            if best_accuracy < test_accuracy:
                best_accuracy = test_accuracy
                best_round = epoch       
        # syn train
        global_model.train()
        global_model.zero_grad()
        
        #ignored_params = list(map(id, global_model.classifier.parameters()))
        #base_params = filter(lambda p: id(p) not in ignored_params,
        #             global_model.parameters())
        #'''
        if args.only_fc == 1:
            for name, param in global_model.named_parameters():
                if 'classifier' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            optimizer = torch.optim.SGD(global_model.classifier.parameters(), lr= 0.01,weight_decay=1e-3)
        else:
            optimizer = torch.optim.SGD(global_model.parameters(), lr= 0.01,weight_decay=1e-3)
       # '''
       # num_ftrs = global_model.classifier.in_features
       
        #global_model.classifier = nn.Linear(num_ftrs, args.num_classes)
        #global_model.classifier = global_model.classifier.to(device)
        global_model.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(global_model.parameters(), lr= 0.01,weight_decay=1e-3)
        #optimizer = torch.optim.SGD(global_model.classifier.parameters(), lr= 0.01,weight_decay=1e-3)
        global_model.zero_grad()
        if epoch<10:
            exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


        '''
        optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': global_model.classifier.parameters(), 'lr': args.lr}
        ], lr=0)
        '''
        #if epoch==0:
        #    syn_all_dataset = copy.deepcopy(syn_round_dataset)
        #else:
        #    syn_all_dataset = torch.utils.data.ConcatDataset([syn_round_dataset,syn_all_dataset])
        syn_all_dataset = copy.deepcopy(syn_round_dataset)
        global_trainloader = torch.utils.data.DataLoader(syn_all_dataset, batch_size=64, shuffle=True)
        
        output_clients = torch.cat(output_clients, dim=0)
        param_augment = dict()
        param_augment['strategy'] = 'crop_gray_brightness_contrast_saturation_hue'

        T = 2.0
        alpha = 0
        #if epoch<3:
        #    alpha=0
        server_epoch=args.server_epoch

        if epoch<10:
            server_epoch=args.server_epoch

        print(len(global_trainloader.dataset))

        for iter in range(server_epoch):
            for batch_idx, (images, labels) in enumerate(global_trainloader):
                images, labels = images.to(device), labels.to(device)
                
                #torch.manual_seed(args.seed+batch_idx)
                #images = augment(images, param_augment)
                images.requires_grad=True
                labels.requires_grad=False
                optimizer.zero_grad()
                output_server = global_model(images)
                loss = criterion(output_server, labels) * (1-alpha)
                #loss += nn.KLDivLoss()(F.log_softmax(output_server/T,dim=1), F.softmax(output_clients/T,dim=1)) * (alpha * T * T) 
                loss.backward()
                #max_norm = 0.25
                #torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm)
                optimizer.step()
                #exp_lr_scheduler.step()
               # print('|Server, Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               #                         epoch + 1, iter + 1, batch_idx * len(images),
                #                        len(global_trainloader.dataset),
                 #                               100. * batch_idx / len(global_trainloader), loss.item()))
        
        tmp_model = global_model.state_dict()

        global_model.load_state_dict(tmp_model)
        # global_model.zero_grad()
        # server_scheduler.step()
        if args.client_decay:
            args.lr = exp_decay_schedule_builder(args.lr, 200, 0.1, True, epoch+1)
        
        for param in global_model.parameters():
            param.requires_grad = True
        # train loss (one round)
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        global_model.eval()
        #if (epoch+1) % 50 ==0:
        #    test_accuracy, test_loss, test_class_acc = test_inference_tsne(args, global_model, test_dataset, device=device)
        #elif epoch==0:
        #    test_accuracy, test_loss, test_class_acc = test_inference_tsne(args, global_model, test_dataset, device=device)
        #else:
        test_accuracy, test_loss, test_class_acc = test_inference(args, global_model, test_dataset, device=device)
        

        test_loss_list.append(test_loss)
        val_acc_list.append(test_accuracy)
        test_class_acc_list.append(test_class_acc)
        # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        loggertxt.info(f' \nAvg Training Stats after {epoch + 1} global rounds:')
        loggertxt.info('Train Accuracy: {:.2f}%'.format(100 * train_accuracy/len(idxs_users)))
        loggertxt.info(f'Training Loss : {loss_avg}')
        loggertxt.info('Test Accuracy: {:.2f}%'.format(100 * test_accuracy))
        loggertxt.info(f'Test Loss: {test_loss} \n')
    
        if best_accuracy < test_accuracy:
            best_accuracy = test_accuracy
            best_round = epoch           
        csvlogger.write_row([test_loss, 100 * test_accuracy])
        if (epoch + 1) % 200 == 0:
            tmp_save_path = os.path.join(save_path_tmp, 'tmp_{}.pt'.format(epoch+1))
            torch.save(global_model.state_dict(),tmp_save_path)
    # Test inference after completion of training
    #test_acc, test_loss, test_class_acc = test_inference(args, global_model, test_dataset, device=device)

    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    if args.hold_normalize:
        client_dict = {}
        for idx, model in enumerate(client_models):
            client_dict['model_{}'.format(idx)] = model.state_dict()
        torch.save(client_dict, SAVE_PATH)
    else:
        torch.save({'global_model': global_model.state_dict()}, SAVE_PATH)

    loggertxt.info(f' \n Results after {args.epochs} global rounds of training:')
    # loggertxt.info("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    loggertxt.info("|---- Best Test Accuracy: {:.2f}% , round: {}".format(100 * best_accuracy, best_round))


    # frac이 1이 아닐경우 잘 작동하지않음.
    # batch_loss_list = np.array(client_loss).sum(axis=0) / args.num_users

    # conv_grad_list = np.array(client_conv_grad).sum(axis=0) / args.num_users
    # fc_grad_list = np.array(client_fc_grad).sum(axis=0) / args.num_users
    # total_grad_list = np.array(client_total_grad_norm).sum(axis=0) /args.num_users
    # client의 avg를 구하고 싶었으나 현재는 client 0만 확인
    # client마다 batch가 다를 경우 bug 예상
    #total_list = [train_loss, val_acc_list, client_loss[0], client_total_grad_norm[0], test_loss_list, client_conv_grad[0], client_fc_grad[0]]
    # train_loss -> 전체 client들의 training loss 평균 per epoch
    # val_acc_list -> 전체 client들의 test acc 평균 per epoch
    # test_loss_list -> test loss 평균 per epoch
    loss_clinet =[]
    total_grad = []
    for i in client_loss[0]:
        loss_clinet.extend(i)
    for i in client_total_grad_norm[0]:
        total_grad.extend(i)
    #total_list = [train_loss, val_acc_list, loss_clinet, total_grad, test_loss_list]
    total_list = [train_loss, total_grad,test_class_acc_list,round_indicate_class,accumlate_indicate_class]

    #total_list =[]
    return total_list,client_indicate_class


def aggregate_scaffold(global_model,server_controls,total_delta_model, total_delta_controls, num_users,idxs_users):
    
    num_of_selected_users = len(idxs_users)
    for delta_controls, delta_models in zip(total_delta_controls,total_delta_model):
        for param, control, del_control, del_model in zip(global_model.parameters(), server_controls,
                                                            delta_controls, delta_models):
            # param.data = param.data + del_model.data * num_of_samples / total_samples / num_of_selected_users
            param.data = param.data + del_model.data / num_of_selected_users
            control.data = control.data + del_control.data / num_users

    return global_model
def match_loss(gw_syn, gw_real, device):
    dis = torch.tensor(0.0).to(device)
    """
    #if args.dis_metric == 'ours':
    for ig in range(len(gw_real)):
        gwr = gw_real[ig]
        gws = gw_syn[ig]
        dis += distance_wb(gwr, gws)
    
    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)
    """
    #elif args.dis_metric == 'cos':
    gw_real_vec = []
    gw_syn_vec = []
    for ig in range(len(gw_real)):
        gw_real_vec.append(gw_real[ig].reshape((-1)))
        gw_syn_vec.append(gw_syn[ig].reshape((-1)))
    gw_real_vec = torch.cat(gw_real_vec, dim=0)
    gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
    dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

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

def update_locals(global_model, local_weights, syn_round_dataset,output_clients,device,test_dataset):

    last_global = copy.deepcopy(global_model)

    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(last_global.parameters(), lr= 0.01, weight_decay=1e-3)
   
    '''
    optimizer = torch.optim.SGD([
        {'params': base_params},
        {'params': global_model.classifier.parameters(), 'lr': args.lr}
    ], lr=0)
    '''
    #if epoch==0:
    #    syn_all_dataset = copy.deepcopy(syn_round_dataset)
    #else:
    #    syn_all_dataset = torch.utils.data.ConcatDataset([syn_round_dataset,syn_all_dataset])
    syn_all_dataset = copy.deepcopy(syn_round_dataset)
    global_trainloader = torch.utils.data.DataLoader(syn_all_dataset, batch_size=128, shuffle=True)
    
    output_clients = torch.cat(output_clients, dim=0)
    param_augment = dict()
    param_augment['strategy'] = 'crop_gray_brightness_contrast_saturation_hue'

    T = 2.0
    alpha = 0
    #if epoch<3:
    #    alpha=0
    server_epoch=0

    new_local_weights = []
    net_parameters = list(global_model.parameters())
    
    print(len(global_trainloader.dataset))
    for local_w in local_weights:
        last_global.load_state_dict(local_w)
        
        optimizer = torch.optim.SGD(last_global.parameters(), lr= 0.01, weight_decay=1e-3)
        local_parameters = list(last_global.parameters())
        print(match_loss(net_parameters,local_parameters,device))
        test_accuracy, test_loss, test_class_acc = test_inference(args, last_global, test_dataset, device=device)
        print("test acc:{}, test loss:{}".format(test_accuracy,test_loss))
        for iter in range(server_epoch):
            for batch_idx, (images, labels) in enumerate(global_trainloader):
                images, labels = images.to(device), labels.to(device)
                #images = augment_batch(images, param_augment)
                images.requires_grad=True
                labels.requires_grad=False
                optimizer.zero_grad()
                output_server = last_global(images)
                loss = criterion(output_server, labels) * (1-alpha)
                #loss += nn.KLDivLoss()(F.log_softmax(output_server/T,dim=1), F.softmax(output_clients/T,dim=1)) * (alpha * T * T) 
                loss.backward()
                #max_norm = 0.25
                #torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm)
                optimizer.step()
                #exp_lr_scheduler.step()
                # print('|Server, Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #                         epoch + 1, iter + 1, batch_idx * len(images),
                #                        len(global_trainloader.dataset),
                    #                               100. * batch_idx / len(global_trainloader), loss.item()))

        new_local_weights.append(last_global.state_dict())
    
    return local_weights

if __name__ == '__main__':
    
    args = easydict.EasyDict({
        "model": 'ConvNet',
        'dataset': 'cifar10',
        'gpu': 0,
        'iid': 0,
        'epochs': 500,
        'optimizer': 'sgd',
        'seed': 0,
        'norm': 'none',
        'num_users': 50,
        'frac': 0.2,
        'local_ep': 10, 
        'local_bs': 32,
        'lr': 0.1,
        'momentum': 0.9,
        'kernel_num': 9,
        'kernel_sizes': '3,4,5',
        'num_channnels': '1',
        'num_filters': 32,
        'max_pool': 'True',
        'num_classes': 10,
        'unequal': 0,
        'stopping_rounds': 10,
        'verbose': 0,
        'hold_normalize': 0,
        'save_path': '../save/checkpoint',
        'exp_folder': 'scaffold_0817',
        'resume': None,
        'server_opt': 'sgd',
        'server_lr': 1,
        'client_decay':0,
        'local_decay':0,
        'alpha': 0.05,
        'server_epoch':0,
        'cosine_norm':0, 
        'only_fc' :0 ,
        'loss':'vic',
        'dc_lr':1,
        'tsne_pred':0 })
    # 실험예정 sgd+m , 10^0.5 , resnet18 재구현, local epoch
    loss_list = [[] for i in range(0, 10)]
    val_acc_list = [[] for i in range(0, 10)]
    batch_loss_list = [[] for i in range(0, 10)]
    conv_grad_list = [[] for i in range(0, 10)]
    fc_grad_list = [[] for i in range(0, 10)]
    total_grad_list = [[] for i in range(0, 10)]
    
    total_list = main_test(args)
    print("total_grad")
    print(total_list[3])
    #train_loss, val_acc_list, batch_loss_list, conv_grad_list, fc_grad_list, total_grad_list = main_test(args)
    #loss_list[0], val_acc_list[0], batch_loss_list[0], conv_grad_list[0], fc_grad_list[0], total_grad_list[0], test_loss_list = main_test(args)
    #print(loss_list[0], val_acc_list[0], batch_loss_list[0], conv_grad_list[0], fc_grad_list[0], total_grad_list[0], test_loss_list )

    # print(np.array(client_loss).sum(axis=0).flatten())

# %%

# %%

# %%

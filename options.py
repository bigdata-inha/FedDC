#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='nothing',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--hold_normalize', type=int, default=0, help='hold Batch Norm or Group Norm, no communication')
    parser.add_argument('--save_path', type=str, default='../save/checkpoint', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    parser.add_argument('--exp_folder', type=str, default='experiment', help='save file name')
    parser.add_argument('--server_opt', type=str, default='sgd', help='server optimizer , [sgd, adam, momentum]')
    parser.add_argument('--server_lr', type=float, default=1.0, help='sever learning rate')
    parser.add_argument('--client_decay', type=int, default=0, help='all client exponetial learning rate decay per round')
    parser.add_argument('--local_decay', type=int, default=0, help='local lr exponetial learning rate decay per epoch')
    parser.add_argument('--alpha', type=float, default=0.1, help='dirichlet distribtution parameter')
    parser.add_argument('--sever_epoch', type=int, default=0, help='sever fine-tuning epoch')
    parser.add_argument('--cosine_norm', type=int, default=0, help='use cosine')
    parser.add_argument('--only_fc', type=int, default=0, help='fine-tuning only fc classifier')
    parser.add_argument('--loss', type=str, default='ce', help='ce, scl, vic')
    parser.add_argument('--dc_lr', type=float, default=1, help='condensation data lr')
    parser.add_argument('--tsne_pred', type=int, default=1, help='pred: 1, label: 0')
    parser.add_argument('--pruning', type=float, default=0.1, help='pruning ratio 0.05 0.1 0.15 0.2')

    
    args = parser.parse_args()

    return args

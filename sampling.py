#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import logging
import random
import torch


#  Settings for a multiplicative linear congruential generator (aka Lehmer
#  generator) suggested in 'Random Number Generators: Good
#  Ones are Hard to Find' by Park and Miller.
MLCG_MODULUS = 2**(31) - 1
MLCG_MULTIPLIER = 16807


# Default quantiles for federated evaluations.
DEFAULT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users, args):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    num_items = int(len(dataset) / num_users)
    # dict_users란? 0~100의 유저들에게 50000개 데이터를 100개씩 할당. 유저마다 indx를 가지고 있는 list
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def imagenet_noniid(dataset, num_users, args, class_num=2):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    #num_shards -> 총클래스 개수/ num_imgs ->한명당 가지는 데이터개수.but imagenet은 클래스마다 다름.세어줘야함 / # idxs 총데이터수
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 아직 한 유저당 클래스 한개만 들어가는 경우 발생.
    
    #idx_shards ->유저당 가지는 랜덤시드 n개(n개는 클래스 개수임.) -> 클래스2 x 유저수100 = 200
    #num_imgs -> 전체데이터셋중 유저 한명이 가지는 한 클래스 데이터 수. 5만/100 =500, 2개클래스 500개
    num_shards, num_imgs = num_users*class_num, int(len(dataset)/num_users/class_num)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs = np.argsort(labels)
    class_count = [0 for i in range(num_shards)]
    for i in labels:
        class_count[i] += 1
    accumulate_class_count =  [0 for i in  range(num_shards)]
    for c in range(num_shards):
        if c==0:
            accumulate_class_count[c] = class_count[0]
        else:
            accumulate_class_count[c] = accumulate_class_count[c-1] + class_count[c]
    idx_shuffle = np.random.permutation(idx_shard)

    client_class_set = []
    for i in range(num_users):
        user_class_set = idx_shuffle[i*class_num:(i+1)*class_num]
        client_class_set.append(user_class_set)
        for class_seed in user_class_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[accumulate_class_count[class_seed] -class_count[class_seed] :accumulate_class_count[class_seed]]), axis=0)        

    return dict_users,client_class_set


def cifar10_iid(train_dataset, num_users, args):

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    n_dataset = len(train_dataset)
    idxs = np.random.permutation(n_dataset)
    batch_idxs = np.array_split(idxs, num_users)
    net_dataidx_map = {i: batch_idxs[i] for i in range(num_users)}

    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts


def partition_data(train_dataset, partition, num_uers, alpha, args):

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_labels = np.array(train_dataset.targets)
    num_train = len(train_dataset)

    if partition == "homo":
        idxs = np.random.permutation(num_train)
        batch_idxs = np.array_split(idxs, num_uers)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_uers)}


    elif partition == "dirichlet":
        min_size = 0
        K = args.num_classes
        N = len(train_labels)   # train data 수 ex)cifar- 50000
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(num_uers)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(train_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_uers))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / num_uers) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_uers):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])
        K = 10
        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_uers)}
            for i in range(10):
                idx_k = np.where(train_labels==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,num_uers)
                for j in range(num_uers):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(10)]
            contain=[]
            for i in range(num_uers):
                current=[i%K]
                times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(num_uers)}
            for i in range(K):
                idx_k = np.where(train_labels==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(num_uers):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1

    traindata_cls_counts = record_net_data_stats(train_labels, net_dataidx_map)
    #print(traindata_cls_counts)

    # return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    # 이전 버전return y_train, net_dataidx_map, traindata_cls_counts
    return net_dataidx_map

def cifar_noniid(dataset, num_users, args, class_num=2):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 아직 한 유저당 클래스 한개만 들어가는 경우 발생.

    #idx_shards ->유저당 갖는 랜덤시드 n개(n개는 클래스 개수임.) -> 클래스2 x 유저수100 = 200
    #num_imgs -> 전체데이터셋중 유저 한명이 가지는 한 클래스 데이터 수. 5만/100 =500, 2개클래스 500개
    num_shards, num_imgs = num_users*class_num, int(len(dataset)/num_users/class_num)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    #sort_index = np.argsort(labels)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    user_classs_dict = []
    # divide and assign
    for i in range(num_users):
        # 200중에 2개 랜덤 선택.
        rand_set = set(np.random.choice(idx_shard, class_num, replace=False))

        if class_num > 1 and i != num_users-1:
            while dataset.targets[idxs[list(rand_set)[1] * num_imgs]] == dataset.targets[idxs[list(rand_set)[0] *num_imgs]]:
                rand_set = set(np.random.choice(idx_shard, class_num, replace=False))
            #print(dataset.targets[idxs[list(rand_set)[1] * num_imgs]])
            #print(dataset.targets[idxs[list(rand_set)[0] * num_imgs]])
            #print('\t')
        user_classs_dict.append(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        #    for data_idx, j in enumerate(dict_users[i]):
        #        print(i, data_idx, dataset.targets[int(j)])
    return dict_users, user_classs_dict


class client_choice(object):
    def __init__(self, args, num_users):
        self.args =args
        self.num_users = num_users
        self.mlcg_start = np.random.RandomState(args.seed).randint(1, MLCG_MODULUS - 1)

    def client_sampling(self, num_users, m, random_seed, round_num):

        #  Settings for a multiplicative linear congruential generator (aka Lehmer
        #  generator) suggested in 'Random Number Generators: Good
        #  Ones are Hard to Find' by Park and Miller.

       
        pseudo_random_int = pow(MLCG_MULTIPLIER, round_num, MLCG_MODULUS) * self.mlcg_start % MLCG_MODULUS
        random_state = np.random.RandomState(pseudo_random_int)
        
        return random_state.choice(num_users, m, replace=False)


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)



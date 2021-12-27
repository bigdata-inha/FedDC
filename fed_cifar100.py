import logging
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

client_ids_train = None
client_ids_test = None
DEFAULT_TRAIN_CLINETS_NUM = 500
DEFAULT_TEST_CLIENTS_NUM = 100
DEFAULT_BATCH_SIZE = 20
DEFAULT_TRAIN_FILE = 'fed_cifar100_train.h5'
DEFAULT_TEST_FILE = 'fed_cifar100_test.h5'

# group name defined by tff in h5 file
_EXAMPLE = 'examples'
_IMGAE = 'image'
_LABEL = 'label'

'''
class BasicDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        img,target = self.x[index], self.y[index]
        return preprocess_cifar_img(img), target
    def __len__(self):
        return len(self.x)

    def cifar100_transform(img_mean, img_std, train = True, crop_size = (32,32)):
        """cropping, flipping, and normalizing."""

        return transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])



    def preprocess_cifar_img(img, train):
        # scale img to range [0,1] to fit ToTensor api
        img = torch.div(img, 255.0)
        transoformed_img = torch.stack([cifar100_transform
            (i.type(torch.DoubleTensor).mean(),
                i.type(torch.DoubleTensor).std(),
                train)
            (i.permute(2,0,1))
            for i in img])
        return transoformed_img


class BasicDataset_Test(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(BasicDataset_Test, self).__init__()

        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        img,target = self.x[index], self.y[index]
        return preprocess_cifar_img(img), target
    def __len__(self):
        return len(self.x)

    def cifar100_transform(img_mean, img_std, train = True, crop_size = (32,32)):
        """cropping, flipping, and normalizing."""

        return transforms.Compose([
               # transforms.ToPILImage(),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])


    def preprocess_cifar_img(img, train):
        # scale img to range [0,1] to fit ToTensor api
        img = torch.div(img, 255.0)
        transoformed_img = torch.stack([cifar100_transform
            (i.type(torch.DoubleTensor).mean(),
                i.type(torch.DoubleTensor).std(),
                train)
            (i.permute(2,0,1))
            for i in img])
        return transoformed_img
'''

def get_dataloader(args, data_dir, train_bs, test_bs, client_idx=None):
    
    # 시드 고정
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TRAIN_FILE), 'r')
    test_h5 = h5py.File(os.path.join(data_dir, DEFAULT_TEST_FILE), 'r')
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    # load data in numpy format from h5 file
    if client_idx is None:
        train_x = np.vstack([train_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in client_ids_train])
        train_y = np.vstack([train_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in client_ids_train]).squeeze()
        test_x = np.vstack([test_h5[_EXAMPLE][client_id][_IMGAE][()] for client_id in client_ids_test])
        test_y = np.vstack([test_h5[_EXAMPLE][client_id][_LABEL][()] for client_id in client_ids_test]).squeeze()
    else:
        client_id_train = client_ids_train[client_idx]
        train_x = np.vstack([train_h5[_EXAMPLE][client_id_train][_IMGAE][()]])
        train_y = np.vstack([train_h5[_EXAMPLE][client_id_train][_LABEL][()]]).squeeze()
        if client_idx <= len(client_ids_test) - 1:
            client_id_test = client_ids_test[client_idx]
            test_x = np.vstack([train_h5[_EXAMPLE][client_id_test][_IMGAE][()]])
            test_y = np.vstack([train_h5[_EXAMPLE][client_id_test][_LABEL][()]]).squeeze()

    # preprocess
    train_x = preprocess_cifar_img(torch.tensor(train_x), train=True)
    train_y = torch.tensor(train_y)
    if len(test_x) != 0:
        test_x = preprocess_cifar_img(torch.tensor(test_x), train=False)
        test_y = torch.tensor(test_y)

    # 필요: train_x,train_y를 list로 저장하고 있기 -> 그때그때 transfrom? 현재 문제 data.tensordataset이용이슈
    # or local 저장
    # generate dataloader
    train_ds = data.TensorDataset(train_x, train_y)
    train_dl = data.DataLoader(dataset=train_ds,
                               batch_size=train_bs,
                               shuffle=True,
                               drop_last=False)

    if len(test_x) != 0:
        test_ds = data.TensorDataset(test_x, test_y)
        test_dl = data.DataLoader(dataset=test_ds,
                                  batch_size=test_bs,
                                  shuffle=True,
                                  drop_last=False)
    else:
        test_dl = None

    train_h5.close()
    test_h5.close()
    return train_dl, test_dl


def load_partition_data_federated_cifar100(args, data_dir, batch_size=DEFAULT_BATCH_SIZE):
    
    # 시드 고정
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    class_num = 100

    # client id list
    train_file_path = os.path.join(data_dir, DEFAULT_TRAIN_FILE)
    test_file_path = os.path.join(data_dir, DEFAULT_TEST_FILE)
    with h5py.File(train_file_path, 'r') as train_h5, h5py.File(test_file_path, 'r') as test_h5:
        global client_ids_train, client_ids_test
        client_ids_train = list(train_h5[_EXAMPLE].keys())
        client_ids_test = list(test_h5[_EXAMPLE].keys())
    
        random.shuffle(client_ids_train)
    # get local dataset
    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()

    for client_idx in range(DEFAULT_TRAIN_CLINETS_NUM):
        train_data_local, test_data_local = get_dataloader(
            args, data_dir, batch_size, batch_size, client_idx)
        local_data_num = len(train_data_local.dataset)
        data_local_num_dict[client_idx] = local_data_num
#        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))
#        logging.info("client_idx = %d, batch_num_train_local = %d" % (client_idx, len(train_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    # global dataset
#    train_data_global = data.DataLoader(
#       data.ConcatDataset(
#            list(dl.dataset for dl in list(train_data_local_dict.values()))
#        ),
#        batch_size=batch_size, shuffle=True)
#    train_data_num = len(train_data_global.dataset)

#    test_data_global = data.DataLoader(
#        data.ConcatDataset(
#            list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None)
#        ),
#        batch_size=batch_size, shuffle=True)
#    test_data_num = len(test_data_global.dataset)
    train_data_global = data.ConcatDataset(
            list(dl.dataset for dl in list(train_data_local_dict.values()))
        )
    test_data_global = data.ConcatDataset(
            list(dl.dataset for dl in list(test_data_local_dict.values()) if dl is not None))

    # DEFAULT_TRAIN_CLINETS_NUM = 500, train_data_num = 50000, test_data_num = 10000
    # train_data_global=5만개 datalodaer, test_data_globa = 1만개 dataloader -> dataset필요시. train_data_global.dataset
    # data_local_num_dict = client data개수 사전, train_data_local_dcit = client dataloader dict,

    return train_data_global, test_data_global, train_data_local_dict,
#    return DEFAULT_TRAIN_CLINETS_NUM, train_data_num, test_data_num, train_data_global, test_data_global, \
#           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num




'''
preprocess reference : https://github.com/google-research/federated/blob/master/utils/datasets/cifar100_dataset.py
'''

def cifar100_transform(img_mean, img_std, train = True, crop_size = (24,24)):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            #transforms.RandomCrop(crop_size),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            #transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack([cifar100_transform
        (i.type(torch.DoubleTensor).mean(),
            i.type(torch.DoubleTensor).std(),
            train)
        (i.permute(2,0,1))
        for i in img])
    return transoformed_img

if __name__ == '__main__':
    data_dir ='../data/fed_cifar100'
    load_partition_data_federated_cifar100(data_dir ,batch_size=20)
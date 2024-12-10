import os
import sys
import torch
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data
import torch.nn.functional as F
from Dataset.init_dataset import Cifar10FL,Cifar100FL,CIFAR_C,CIFAR10_C_origin, CIFAR100_RandomC
from torch.autograd import Variable
from Network.Models_Def.resnet import ResNet10,ResNet12
from Network.Models_Def.shufflenet import ShuffleNetG2
from Network.Models_Def.mobilnet_v2 import MobileNetV2
import torchvision.transforms as transforms
import random
from Dataset.dataaug import AugMixDataset, AugMixPublicDataset
from torchvision import datasets

Seed = 0
seed = Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
Project_Path = r'/home/fangxiuwen/'

def init_logs(log_level=logging.INFO,log_path = Project_Path+'Logs/',sub_name=None):
    # logging：https://www.cnblogs.com/CJOKER/p/8295272.html
    # 第一步，创建一个logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    log_path = log_path
    mkdirs(log_path)
    filename = os.path.basename(sys.argv[0][0:-3])
    if sub_name == None:
        log_name = log_path + filename + '.log'
    else:
        log_name = log_path + filename + '_' + sub_name +'.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level)  # 输出到file的log等级的开关
    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    console  = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    # 第四步，将logger添加到handler里面
    logger.addHandler(fh)
    logger.addHandler(console)
    # 日志
    return logger

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_cifar10_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    # noise_type = 'pairflip' #[pairflip, symmetric]
    # noise_rate = 0.1
    cifar10_train_ds = Cifar10FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    cifar10_test_ds = Cifar10FL(datadir, train=False, download=True, transform=transform)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_train_ds = Cifar100FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    cifar100_test_ds = Cifar100FL(datadir, train=False, download=True, transform=transform)
    X_train, y_train = cifar100_train_ds.data, cifar100_test_ds.target
    X_test, y_test = cifar100_train_ds.data, cifar100_test_ds.target
    return (X_train, y_train, X_test, y_test)


def generate_public_data_indexs(dataset,datadir,size, noise_type=None, noise_rate=0):
    if dataset =='cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    n_train = y_train.shape[0]
    idxs = np.random.permutation(n_train)
    idxs = idxs[0:size]
    return idxs

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, corrupt_type=None, corrupt_rate=0, test_dataset=None, test_corrupt_rate=0):
    #For the train dataset:
    if dataset == 'cifar10':
        dl_obj = Cifar10FL
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            #Augmix取消后两行
            transforms.ToTensor(),
            normalize
        ])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)

    if dataset =='cifar100':
        dl_obj=Cifar100FL
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)

    if dataset == 'cifar10c':
        dl_obj = CIFAR_C
        normalize_train = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1944, 0.2010])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize_train
        ])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, c_type=corrupt_type, transform=transform_train, corrupt_rate=corrupt_rate)

    # For the test dataset:
    if test_dataset == 'clean':
        test_datadir = '../Dataset/cifar_10'
        normalize_test = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_test
        ])
        test_ds = Cifar10FL(test_datadir, train=False, transform=transform_test, download=False)
    elif test_dataset == 'corrupt':
        test_datadir = '../Dataset/cifar_10/CIFAR-10-C_test'
        normalize_test = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2023, 0.1944, 0.2010])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_test
        ])
        test_ds = CIFAR_C(test_datadir, transform=transform_test, corrupt_rate=test_corrupt_rate)


    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, drop_last=True, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds

def get_augmix_private_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, corrupt_type=None, corrupt_rate=0, test_dataset=None, test_corrupt_rate=0):
    #For the train dataset:
    if dataset == 'cifar10':
        dl_obj = Cifar10FL
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            #Augmix取消后两行
            # transforms.ToTensor(),
            # normalize
        ])
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)
        train_ds = AugMixDataset(train_ds, preprocess, no_jsd=False)

    if dataset =='cifar100':
        dl_obj=Cifar100FL
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=False)

    if dataset == 'cifar10c':
        dl_obj = CIFAR_C
        normalize_train = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1944, 0.2010])
        # transform_train = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize_train
        # ])
        #Augmix transform
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4), mode='reflect').data.squeeze()),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=noise_level),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip()
        ])
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, c_type=corrupt_type, transform=transform_train, corrupt_rate=corrupt_rate)
        train_ds = AugMixDataset(train_ds, preprocess, no_jsd=False)

    # For the test dataset:
    if test_dataset == 'clean':
        test_datadir = '../Dataset/cifar_10'
        normalize_test = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                              std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_test
        ])
        test_ds = Cifar10FL(test_datadir, train=False, transform=transform_test, download=False)
    elif test_dataset == 'corrupt':
        test_datadir = '../Dataset/cifar_10/CIFAR-10-C_test'
        normalize_test = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1944, 0.2010])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_test
        ])
        test_ds = CIFAR_C(test_datadir, transform=transform_test, corrupt_rate=test_corrupt_rate)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True, num_workers=8)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, drop_last=True, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds

def get_augmixcorrupt_randompub_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, test_dataset=None, test_corrupt_rate=0):
    #For the train dataset:
    if dataset =='cifar100':
        # corrupt cifar100
        dl_obj=CIFAR_C
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            # transforms.ToTensor(),
            # normalize
        ])
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)])
        train_ds = dl_obj(datadir, dataidxs=dataidxs, transform=transform_train, corrupt_rate=1)
        train_ds = AugMixPublicDataset(train_ds, preprocess, corrupt='augmix')

    # For the test dataset:
    if test_dataset == 'clean':
        test_datadir = '../Dataset/cifar_10'
        normalize_test = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                              std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_test
        ])
        test_ds = Cifar10FL(test_datadir, train=False, transform=transform_test, download=False)
    elif test_dataset == 'corrupt':
        test_datadir = '../Dataset/cifar_10/CIFAR-10-C_test'
        normalize_test = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1944, 0.2010])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_test
        ])
        test_ds = CIFAR_C(test_datadir, transform=transform_test, corrupt_rate=test_corrupt_rate)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, drop_last=True, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds


def init_nets(n_parties,nets_name_list, num_classes = 10):
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name=='ResNet10':
            net = ResNet10(num_classes=num_classes)
        elif net_name =='ResNet12':
            net = ResNet12(num_classes=num_classes)
        elif net_name =='ShuffleNet':
            net = ShuffleNetG2(num_classes=num_classes)
        elif net_name =='Mobilenetv2':
            net = MobileNetV2(num_classes=num_classes)
        nets_list[net_i] = net
    return nets_list

if __name__ =='__main__':
    logger = init_logs()
    public_data_indexs = generate_public_data_indexs(dataset='cifar100',datadir='./cifar_100',size=5000)
    train_dl, test_dl, train_ds, test_ds = get_dataloader(dataset='cifar100',datadir='./cifar_100',train_bs=256,test_bs=512,dataidxs=public_data_indexs)
    print(len(train_ds))
    # loss_list = [[1,2,3,4,5],[2,3,4,5,6]]
    # loss_name = 'test'
    # draw_epoch_loss(loss_list,loss_name,savepath='./sda.png')
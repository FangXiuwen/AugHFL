import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"
import sys
sys.path.append("..")
from Dataset.utils import init_logs, get_dataloader, init_nets, mkdirs
from Network.Models_Def.resnet import ResNet10,ResNet12
from matplotlib.colors import ListedColormap
from loss import SCELoss
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import manifold
import torch.nn as nn
import numpy as np
from numpy import *
import random
import torch
import torch.backends.cudnn
from Dataset.tinyimagenet_dataloader import get_tinyimagenet_dataloader

'''
Global Parameters
'''
Seed = 0
N_Participants = 4
TrainBatchSize = 256
TestBatchSize = 512
Pretrain_Epoch = 40 #20
Private_Data_Len = 25000
# Private_Data_Len = 50000
Pariticpant_Params = {
    'loss_funnction' : 'CE',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.01
}

"""Corruption Setting"""
Corruption_Type = 'clean' #['clean', 'random_noise']
Corrupt_rate = 0 #[0, 1, 0.5]
Test_Corruption_Type = 'clean' #['clean', 'random_noise']
Test_Corrupt_rate = 0#[0, 1]
if Test_Corrupt_rate == 0:
    test_dataset = 'clean'
else:
    test_dataset = 'corrupt'
"""Noise Setting"""
Noise_type = None #['pairflip','symmetric',None]
Noise_rate = 0
"""Heterogeneous Model Setting"""
Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']
"""Homogeneous Model Setting"""
#Nets_Name_List = ['ResNet12','ResNet12','ResNet12','ResNet12']
"""Dataset Setting"""
Dataset_Name = 'tinyimagenet'
Dataset_Dir = '../Dataset/tiny_imagenet'

Dataset_Classes = [i for i in range(200)]
Output_Channel = len(Dataset_Classes)

# Dataset_Name = 'cifar10'
# Dataset_Dir = '../Dataset/cifar_10'
# Dataset_Classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Output_Channel = len(Dataset_Classes)


def pretrain_network(epoch,net,train_dataloader,test_dataloader,loss_function,optimizer_name,learning_rate):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='SCE':
        criterion = SCELoss(alpha=0.1, beta=1, num_classes=10)
    criterion.to(device)
    if optimizer_name =='Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if optimizer_name =='SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    for _epoch in range(epoch):
        log_interval = 100
        # correct = 0
        # total = 0
        net.train()
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = net(images)
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            labels = labels.long()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    _epoch, batch_idx * len(images), len(train_dataloader.dataset),
                            100. * batch_idx / len(train_dataloader), loss.item()))
        scheduler.step()
        net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs,_ = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            logger.info('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return net

def evaluate_network(net,dataloader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        logger.info('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    return 100 * correct / total

if __name__ =='__main__':
    mkdirs('./Model_Storage_tinyimagenet/' + Corruption_Type + '_' + str(Corrupt_rate) + '_' + Pariticpant_Params['loss_funnction'])
    # mkdirs('./Model_Storage/' + Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) +str(Noise_rate))
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device_ids = [0,1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Load Participants' Data and Model")

    net_dataidx_map = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(100000)
        idxes = idxes[0:Private_Data_Len]
        net_dataidx_map[index]= idxes
    logger.info(net_dataidx_map)
    net_list = init_nets(n_parties=N_Participants,nets_name_list=Nets_Name_List,num_classes = Output_Channel)

    test_accuracy_list = []

    for index in range(N_Participants):
        logger.info('Pretrain Participants Model {}'.format(index))
        train_dl_local ,test_dl, train_ds_local, test_ds = get_tinyimagenet_dataloader(dataset=Dataset_Name,datadir=Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=net_dataidx_map[index],
                                                                        corrupt_type=Corruption_Type, corrupt_rate=Corrupt_rate,
                                                                        test_dataset=test_dataset, test_corruption_type=Test_Corruption_Type, test_corrupt_rate=Test_Corrupt_rate)
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]
        logger.info('Pretrain the '+str(index)+' th Participant Model with N_training: '+str(len(train_ds_local)))
        network = pretrain_network(epoch=Pretrain_Epoch,net=network,train_dataloader=train_dl_local,test_dataloader=test_dl,loss_function=Pariticpant_Params['loss_funnction'],optimizer_name=Pariticpant_Params['optimizer_name'],learning_rate=Pariticpant_Params['learning_rate'])
        logger.info('Save the '+str(index)+' th Participant Model')
        torch.save(network.state_dict(), './Model_Storage_tinyimagenet/' + Corruption_Type + '_' + str(Corrupt_rate)+ '_' + Pariticpant_Params['loss_funnction'] + '/'+netname+'_'+str(index)+'.ckpt')
        # torch.save(network.state_dict(), './Model_Storage/' +Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) + str(Noise_rate)+ '/'+netname+'_'+str(index)+'.ckpt')

    for index in range(N_Participants):
        logger.info('Evaluate Model {}'.format(index))
        train_dl, test_dl, _, _ = get_tinyimagenet_dataloader(dataset=Dataset_Name,datadir=Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=net_dataidx_map[index],
                                        corrupt_type=Corruption_Type, corrupt_rate=Corrupt_rate,
                                        test_dataset=test_dataset, test_corruption_type=Test_Corruption_Type, test_corrupt_rate=Test_Corrupt_rate)
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]
        network.load_state_dict(torch.load('./Model_Storage_tinyimagenet/' + Corruption_Type + '_' + str(Corrupt_rate)+ '_' + Pariticpant_Params['loss_funnction'] + '/'+netname+'_'+str(index)+'.ckpt'))
        # network.load_state_dict(torch.load('./Model_Storage/'+Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) + str(Noise_rate)+ '/'+netname+'_'+str(index)+'.ckpt'))
        output = evaluate_network(net=network,dataloader=test_dl)
        test_accuracy_list.append(output)
    print('The average Accuracy of models on the test images:'+str(mean(test_accuracy_list)))

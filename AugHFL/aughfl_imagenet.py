import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
import sys
sys.path.append("..")
from Dataset.utils import init_logs, get_dataloader, get_augmix_private_dataloader, get_augmixcorrupt_randompub_dataloader, init_nets, generate_public_data_indexs, mkdirs
from loss import SCELoss
import torch.nn.functional as F
import torch.optim as optim
from random import sample
import torch.nn as nn
import numpy as np
from numpy import *
import random
import torch
import torch.backends.cudnn
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from Dataset.tinyimagenet_dataloader import get_augmix_tinyimagenet_dataloader

'''
Global Parameters
'''
Seed = 0
N_Participants = 4 #10
TrainBatchSize = 256 # 256
TestBatchSize = 512
CommunicationEpoch = 40
Pariticpant_Params = {
    'loss_funnction' : 'CE',
    'optimizer_name' : 'Adam',
    'learning_rate' : 0.001

}

"""Corruption Setting"""
Corruption_Type = 'random_noise' #['clean', 'random_noise']
Corrupt_rate = 1 #[0, 1, 0.5]
Test_Corruption_Type = 'random_noise' #['clean', 'random_noise']
Test_Corrupt_rate = 1 #[0, 1]
if Test_Corrupt_rate == 0:
    test_dataset = 'clean'
else:
    test_dataset = 'corrupt'
"""Noise Setting"""
Noise_type = None #['pairflip','symmetric',None]
Noise_rate = 0
"""Heterogeneous Model Setting"""
Private_Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']
"""Homogeneous Model Setting"""
# Private_Nets_Name_List = ['ResNet12','ResNet12','ResNet12','ResNet12']
"""Dataset Setting"""
Private_Dataset_Name = 'tinyimagenet'
Private_Dataset_Dir = '../Dataset/tiny_imagenet'
# Private_Data_Len = 10000
Private_Data_Len = 25000
Private_Dataset_Classes = [i for i in range(200)]
Private_Output_Channel = len(Private_Dataset_Classes)
"""Public Dataset Setting"""
Public_Dataset_Name = 'cifar100'
Public_Dataset_Dir = '../Dataset/cifar_100/CIFAR-100-C_train'
Public_Dataset_Length = 5000

def evaluate_network(network,dataloader,logger):
    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        logger.info('Test Accuracy of the model on the test images: {} %'.format(acc))
    return acc

def update_model_via_private_data(network,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='SCE':
        criterion = SCELoss(alpha=0.1, beta=1.0, num_classes=10)

    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    if optimizer_method =='SGD':
        optimizer = optim.SGD(network.parameters(), lr=learing_rate, momentum=0.9, weight_decay=1e-4)
    participant_local_loss_batch_list = []
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=2e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32)
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            #Augmix+JSD
            images_all = torch.cat(images, 0).cuda()
            labels = labels.to(device)
            logits_all,_ = network(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, images[0].size(0))
            # Cross-entropy is only computed on clean images
            loss = criterion(logits_clean, labels.long())
            # loss = criterion(logits_clean, labels)
            p_clean, p_aug1, p_aug2 = F.softmax(
                logits_clean, dim=1), F.softmax(
                logits_aug1, dim=1), F.softmax(
                logits_aug2, dim=1)
            # Clamp mixture distribution to avoid exploding KL divergence
            p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
            loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                          F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(images[0]), len(private_dataloader.dataset),
                    100. * batch_idx / len(private_dataloader), loss.item()))

    return network,participant_local_loss_batch_list



if __name__ =='__main__':
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
    device_ids = [0,1,2,3,4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Initialize Participants' Data idxs and Model")
    net_dataidx_map = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(100000)
        idxes = idxes[0:Private_Data_Len]
        net_dataidx_map[index]= idxes
    logger.info(net_dataidx_map)

    net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Nets_Name_List,num_classes = Private_Output_Channel)
    logger.info("Load Participants' Models")

    for i in range(N_Participants):
        network = net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Nets_Name_List[i]
        network.load_state_dict(torch.load('../Network/Model_Storage_tinyimagenet/' + Corruption_Type + '_' + str(Corrupt_rate)+ '_' + Pariticpant_Params['loss_funnction'] + '/' + netname + '_' +str(i)+'.ckpt'))
        # network.load_state_dict(torch.load('../Network/Model_Storage/' + Pariticpant_Params['loss_funnction'] + '/' + str(Noise_type) + str(Noise_rate)+ '/' + netname + '_' + str(i) + '.ckpt'))

        # #For evaluate
        # network.load_state_dict(torch.load('./test/Model_Storage_tinyimagenet/' + Corruption_Type + '_' + str(Corrupt_rate)+ '_CE/' + netname + '_' +str(i)+'.ckpt'))


    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_indexs(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,size=Public_Dataset_Length, noise_type=Noise_type, noise_rate=Noise_rate)
    public_train_dl, _, public_train_ds, _ = get_augmixcorrupt_randompub_dataloader(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,
                                                                          train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=public_data_indexs,
                                                                          test_dataset=test_dataset, test_corruption_type=Test_Corruption_Type,
                                                                          test_corrupt_rate=Test_Corrupt_rate)

    col_loss_list = []
    local_loss_list = []
    acc_list = []
    for epoch_index in range(CommunicationEpoch):
        logger.info("The "+str(epoch_index)+" th Communication Epoch")

        logger.info('Evaluate Models')
        acc_epoch_list = []
        for participant_index in range(N_Participants):
            netname = Private_Nets_Name_List[participant_index]
            private_dataset_dir = Private_Dataset_Dir
            # print(netname + '_' + Private_Dataset_Name + '_' + private_dataset_dir)
            _, test_dl, _, _ = get_augmix_tinyimagenet_dataloader(dataset=Private_Dataset_Name, datadir=private_dataset_dir, train_bs=TrainBatchSize,
                                              test_bs=TestBatchSize, dataidxs=None,
                                              corrupt_type=Corruption_Type, corrupt_rate=Corrupt_rate,
                                              test_dataset=test_dataset, test_corruption_type=Test_Corruption_Type,
                                              test_corrupt_rate=Test_Corrupt_rate)
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
            acc_epoch_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / N_Participants
        logger.info('Avg Accuracy:{}'.format(accuracy_avg))

        '''
        HHF
        '''
        for batch_idx, (images, _) in enumerate(public_train_dl):
            linear_output_list = []
            linear_output_target_list = []
            kl_loss_batch_list = []
            participant_kl_list = []
            '''
            Calculate Linear Output
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                image = images[participant_index]
                images_all = torch.cat(image, 0).cuda()
                logits_all, _ = network(x=images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, image[0].size(0))
                p_clean, p_aug1, p_aug2 = F.softmax(
                    logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                    logits_aug2, dim=1)
                plog_clean = p_clean.log()
                linear_output_target_list.append(p_clean.clone().detach())
                linear_output_list.append(plog_clean)
                participant_kl = 1 / (F.kl_div(p_aug1.log(), p_clean, reduction='batchmean') + F.kl_div(p_aug2.log(), p_clean, reduction='batchmean'))

                # p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                # participant_kl = 1 / (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                #             F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                #             F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
                participant_kl_list.append(participant_kl.clone().detach().cpu())

            #归一化
            participant_weight_list = []
            kl_sum = sum(participant_kl_list)
            for participant_index in range(N_Participants):
                participant_weight_list.append(participant_kl_list[participant_index] / kl_sum)

            '''
            Update Participants' Models via KL Loss and Data Quality
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                criterion = nn.KLDivLoss(reduction='batchmean')
                criterion.to(device)
                optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                optimizer.zero_grad()
                loss = torch.tensor(0)
                for i in range(N_Participants):
                    if i != participant_index:
                        # for HFL+PubAug
                        weight_index = participant_weight_list[i]
                        # # for solo HFL
                        # weight_index = 1 / (N_Participants - 1)
                        loss_batch_sample = criterion(linear_output_list[participant_index], linear_output_target_list[i])
                        temp = weight_index * loss_batch_sample
                        loss = loss + temp
                kl_loss_batch_list.append(loss.item())
                loss.backward()
                optimizer.step()
            col_loss_list.append(kl_loss_batch_list)

        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl_local, _, train_ds_local, _ = get_augmix_tinyimagenet_dataloader(dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                                                                  train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                                  dataidxs=private_dataidx, corrupt_type=Corruption_Type,
                                                                  corrupt_rate=Corrupt_rate, test_dataset=test_dataset,
                                                                  test_corruption_type=Test_Corruption_Type, test_corrupt_rate=Test_Corrupt_rate)
            private_epoch = max(int(len(public_train_ds)/len(train_ds_local)),1)

            network,private_loss_batch_list = update_model_via_private_data(network=network,private_epoch=private_epoch,
            private_dataloader=train_dl_local,loss_function=Pariticpant_Params['loss_funnction'],
            optimizer_method=Pariticpant_Params['optimizer_name'],learing_rate=Pariticpant_Params['learning_rate'],
            logger=logger)
            mean_privat_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_privat_loss_batch)
        local_loss_list.append(local_loss_batch_list)

        """
        Evaluate ModelS in the final round
        """
        if epoch_index == CommunicationEpoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):  # 改成2 拿来测试 N_Participants
                _, test_dl, _, _ = get_augmix_tinyimagenet_dataloader(dataset=Private_Dataset_Name, datadir=Private_Dataset_Dir,
                                                  train_bs=TrainBatchSize,
                                                  test_bs=TestBatchSize, dataidxs=None,
                                                  corrupt_type=Corruption_Type, corrupt_rate=Corrupt_rate,
                                                  test_dataset=test_dataset, test_corruption_type=Test_Corruption_Type,
                                                  test_corrupt_rate=Test_Corrupt_rate)
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
                acc_epoch_list.append(accuracy)
            acc_list.append(acc_epoch_list)
            accuracy_avg = sum(acc_epoch_list) / N_Participants
            logger.info('Avg accuracy:{}'.format(accuracy_avg))

        if epoch_index % 5 == 0 or epoch_index==CommunicationEpoch-1:
            mkdirs('./test/Performance_Analysis/' + Corruption_Type + '_' + str(Corrupt_rate))
            mkdirs('./test/Model_Storage_tinyimagenet/' + Corruption_Type + '_' + str(Corrupt_rate)+ '_' + Pariticpant_Params['loss_funnction'])

            logger.info('Save Loss')
            col_loss_array = np.array(col_loss_list)
            np.save('./test/Performance_Analysis/' + Corruption_Type + '_' + str(Corrupt_rate)
                    +'/collaborative_loss.npy', col_loss_array)
            local_loss_array = np.array(local_loss_list)
            np.save('./test/Performance_Analysis/' + Corruption_Type + '_' + str(Corrupt_rate)
                    +'/local_loss.npy', local_loss_array)
            logger.info('Save Acc')
            acc_array = np.array(acc_list)
            np.save('./test/Performance_Analysis/' + Corruption_Type + '_' + str(Corrupt_rate)
                    +'/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(N_Participants):
                netname = Private_Nets_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(),
                           './test/Model_Storage_tinyimagenet/' + Corruption_Type + '_' + str(Corrupt_rate) + '_' + Pariticpant_Params['loss_funnction']+ '/' + netname + '_' +str(participant_index)+'.ckpt')

        # if epoch_index % 5 == 0 or epoch_index==CommunicationEpoch-1:
        #     mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction'])
        #     mkdirs('./test/Model_Storage/' +Pariticpant_Params['loss_funnction'])
        #     mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+str(Noise_type))
        #     mkdirs('./test/Model_Storage/'+Pariticpant_Params['loss_funnction']+str(Noise_type))
        #     mkdirs('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+'/'+str(Noise_type)+str(Noise_rate))
        #     mkdirs('./test/Model_Storage/' +Pariticpant_Params['loss_funnction']+'/'+str(Noise_type)+ str(Noise_rate))
        #
        #     logger.info('Save Loss')
        #     col_loss_array = np.array(col_loss_list)
        #     np.save('./test/Performance_Analysis/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) +str(Noise_rate)
        #             +'/collaborative_loss.npy', col_loss_array)
        #     local_loss_array = np.array(local_loss_list)
        #     np.save('./test/Performance_Analysis/'+Pariticpant_Params['loss_funnction']+'/'+str(Noise_type) +str(Noise_rate)
        #             +'/local_loss.npy', local_loss_array)
        #     logger.info('Save Acc')
        #     acc_array = np.array(acc_list)
        #     np.save('./test/Performance_Analysis/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) +str(Noise_rate)
        #             +'/acc.npy', acc_array)
        #
        #     logger.info('Save Models')
        #     for participant_index in range(N_Participants):
        #         netname = Private_Nets_Name_List[participant_index]
        #         network = net_list[participant_index]
        #         network = nn.DataParallel(network, device_ids=device_ids).to(device)
        #         torch.save(network.state_dict(),
        #                    './test/Model_Storage/' +Pariticpant_Params['loss_funnction']+'/'+ str(Noise_type) + str(Noise_rate) + '/'
        #                    + netname + '_' + str(participant_index) + '.ckpt')


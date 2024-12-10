import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
"""
Origin data
"""
# train_data = dset.CIFAR10('../Dataset/cifar_10', train=True)
# for img, label in zip(train_data.data, train_data.targets):
#     plt.cla()
#     plt.imshow(np.transpose(img, (0, 1, 2)))
#     plt.pause(0.1)
#     label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#     print(label_name[label])

"""
Corrupt data
"""
data = np.load('cifar_100/CIFAR-100-C_train/random_noise.npy')
label= np.load('cifar_100/CIFAR-100-C_train/labels.npy')
# data = data.reshape((50000, 3, 32, 32))

for j in range(50000):
    imgdata = data[j, :, :, :]
    plt.cla()
    plt.imshow(np.transpose(imgdata, (0, 1, 2)))
    plt.pause(0.1)
    label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # print(label_name[label[j]])

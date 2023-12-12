# AugHFL

This repository provides resources for the following paper:

> [**Robust Heterogeneous Federated Learning under Data Corruption**]([ICCV 2023 Open Access Repository](https://openaccess.thecvf.com/content/ICCV2023/html/Fang_Robust_Heterogeneous_Federated_Learning_under_Data_Corruption_ICCV_2023_paper.html))  
> Xiuwen Fang, [Mang Ye](https://marswhu.github.io/index.html), Xiyuan Yang *ICCV 2023*

# [AugHFL Description](#contents)

AugHFL(Augmented Heterogeneous Federated Learning) is a federated learning framework to investigate the problem of data corruption in the model heterogeneous federated learning:

1. Local Learning with Data Corruption.

2. Robust Corrupted Clients Communication.

# [Framework Architecture](#contents)

![](C:\Users\PC\AppData\Roaming\marktext\images\2023-12-12-13-18-52-image.png)

# [Dataset](#contents)

Our experiments are conducted on two datasets, Cifar-10-C and Cifar-100. We set public dataset on the server as a subset of Cifar-100, and randomly divide Cifar-10-C to different clients as private datasets.

Dataset used: [CIFAR-10-C](https://zenodo.org/records/2535967)、[CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar.html)

Note: Data will be processed in init_data.py

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# init public data and local data
python Dataset/init_data.py
# pretrain local models
python Network/pretrain.py
# AugHFL
python AugHFL/AugHFL.py
```

# [Script and Sample Code](#contents)

```bash
├── Robust_FL
    ├── Dataset
        ├── augmentations.py
        ├── cifar.py
        ├── dataaug.py
        ├── init_dataset.py
        ├── utils.py
    ├── Network
        ├── Models_Def
            ├── mobilnet_v2.py
            ├── resnet.py
            ├── shufflenet.py
        ├── pretrain.py
    ├── AugHFL
        ├── AugHFL.py
    ├── loss.py
    ├── README.md
```

# [Comparison with the SOTA methods](#contents)

In the heterogeneous model scenario, we assign four different networks:ResNet10,ResNet12,ShuffleNet,Mobilenetv2

![](C:\Users\PC\AppData\Roaming\marktext\images\2023-12-12-13-12-42-image.png)

# [Citation](#contents)

```citation
@inproceedings{fang2023robust,
  title={Robust heterogeneous federated learning under data corruption},
  author={Fang, Xiuwen and Ye, Mang and Yang, Xiyuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5020--5030},
  year={2023}
}
```

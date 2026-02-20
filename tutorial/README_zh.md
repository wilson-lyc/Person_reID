# UMacau-Person-reID-Practical
[![Readme-EN](https://img.shields.io/badge/README-English-green.svg)](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/tutorial)    [![Readme-CN](https://img.shields.io/badge/README-中文-green.svg)](https://zhuanlan.zhihu.com/p/50387521)

By [Zhedong Zheng](http://zdzheng.xyz/)

这是[澳门大学](https://www.cis.um.edu.mo/)的计算机视觉实践课，由Zhedong Zheng编写。
本实践课将探索行人特征学习的基础。我们将一步步学习构建一个简单的人员重识别系统。(8分钟阅读) :+1: **欢迎任何建议。**

![](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/show.webp)

人员重识别可以被视为一个图像检索问题。给定摄像机**A**中的一张查询图像，我们需要在其他摄像机中找到同一人的图像。人员重识别的关键是找到有区分度的人员表示。许多最近的研究采用深度学习模型来提取视觉特征，并取得了最先进的性能。

我们可以利用这项技术来帮助人们。查看Nvidia的精彩视频。(https://youtu.be/GiZ7kyrwZGQ?t=60)

## 关键词
Person re-identification, 行人重识别, 人の再識別, 보행자 재 식별, Réidentification des piétons, Ri-identificazione pedonale, Fußgänger-Neuidentifizierung, إعادة تحديد المشاة, Re-identificación de peatones

## Ubuntu 使用方法
**如果机器已安装cuda工具包和nvidia驱动（如我们学校的台式机），在本教程中不需要任何root权限，如sudo。如果使用自己的机器，我建议先参考这个答案安装cuda。https://askubuntu.com/questions/1077061/how-do-i-install-nvidia-and-cuda-drivers-into-ubuntu**

假设你有一个Ubuntu桌面系统，

按 `Ctrl+Alt+T` 打开新终端。

默认路径类似 `\home\user`。

然后可以输入 `ls` 列出所有子文件夹。
```bash
ls
```
可能会显示类似
```bash
Desktop Images Musics Downloads ...
```

然后可以输入 `cd XXX` 进入子文件夹，例如
```bash
cd Downloads # 如果你下载的数据集在Downloads文件夹中。
ls # 显示 \home\user\Downloads 中的所有内容
cd ..   #返回上级文件夹。回到 \home\user
```

## Windows 使用方法（不推荐）
考虑到较低的GPU利用率和意外错误，我们不建议使用Windows。
如果你仍想使用Windows，需要注意两点。
- 路径：Ubuntu路径是 `\home\zzd\` 但Windows路径是 `D://Downloads/`，使用 `/` 而不是 `\`
- 多线程：Pytorch（Windows版本）不支持多线程读取数据。训练和测试时请设置 `num_workers=0`
- 无Triton或其他错误：请移除[训练代码行](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/train.py#L512)和[测试代码行](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/test.py#L167)中的 `torch.compile`。

另请参阅 https://github.com/layumi/Person_reID_baseline_pytorch/issues/34

## Colab 使用方法（不推荐）
请参阅 https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/colab

## 前提条件
- 下载我的代码仓库
```bash
git clone https://github.com/layumi/Person_reID_baseline_pytorch.git # 下载全部代码。
cd Person_reID_baseline_pytorch
```
- 从 http://pytorch.org/ 安装Pytorch
- 安装所需包
```bash
pip install -r requirements.txt
```
- [可选] 没有pip或python？？你可以通过安装miniconda来无需sudo权限安装：
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
- [可选] 你可以跳过这一步。通常它会随pytorch一起安装。从源码安装Torchvision
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [可选] 你可以跳过这一步。从源码安装apex
```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
因为pytorch和torchvision是正在开发中的项目。

这里我们注意到我们的代码是基于Pytorch 0.3.0/0.4.0/0.5.0/1.0.0和Torchvision 0.2.0/0.2.1测试的。
在大多数情况下，我们也支持最新的pytorch。我们通常建议使用最新的pytorch。

## 入门
查看前提条件。本实践的下载链接如下：

- 代码：[ReID-Baseline](https://github.com/layumi/Person_reID_baseline_pytorch)
- 数据：[Market-1501](http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[百度]](https://pan.baidu.com/s/1ntIi2Op)

下载Market-1501的快速命令行：
```
pip install gdown
gdown https://drive.google.com/uc\?id\=0B8-rUzbwVRk0c054eEozWG9COHM
```

## 第一部分：训练
### 第一部分1：准备数据文件夹（`python prepare.py`）
你可能注意到下载的文件夹结构如下：
```
├── Market/
│   ├── bounding_box_test/          /* 测试用文件（候选图像库）
│   ├── bounding_box_train/         /* 训练用文件
│   ├── gt_bbox/                    /* 多查询测试用文件
│   ├── gt_query/                   /* 我们不使用它
│   ├── query/                      /* 测试用文件（查询图像）
│   ├── readme.txt
```
在编辑器中打开并编辑脚本 `prepare.py`。将 `prepare.py` 中的第五行改为你下载的路径，例如 `\home\zzd\Download\Market`。在终端中运行此脚本。
```bash
python prepare.py
```
我们在下载文件夹下创建一个名为 `pytorch` 的子文件夹。
```
├── Market/
│   ├── bounding_box_test/          /* 测试用文件（候选图像库）
│   ├── bounding_box_train/         /* 训练用文件
│   ├── gt_bbox/                    /* 多查询测试用文件
│   ├── gt_query/                   /* 我们不使用它
│   ├── query/                      /* 测试用文件（查询图像）
│   ├── readme.txt
│   ├── pytorch/
│       ├── train/                   /* 训练
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* 验证
│       ├── train_all/               /* 训练+验证
│       ├── query/                   /* 查询文件
│       ├── gallery/                 /* 图库文件
│       ├── multi-query/
```

在每个子目录中，例如 `pytorch/train/0002`，相同ID的图像被放在同一个文件夹中。
现在我们已成功为 `torchvision` 准备好数据。

```diff
+ Quick Question. 如何识别相同ID的图像？
```
对于Market-1501，图像名包含身份标签和摄像机ID。请在此处查看命名规则[here](http://www.liangzheng.org/Project/project_reid.html)。

对于DukeMTMC，你可以使用我修改的 `python prepare_Duke.py`。

### 第一部分2：构建神经网络（`model.py`）
我们可以使用预训练网络，如 `AlexNet`、`VGG16`、`ResNet` 和 `DenseNet`。通常，预训练网络有助于获得更好的性能，因为它保留了ImageNet中的一些好的视觉模式[1]。

在pytorch中，我们只需两行代码就可以轻松导入它们。例如，
```python
from torchvision import models
model = models.resnet50(pretrained=True)
```
你可以简单地通过以下方式检查模型结构：
```python
print(model)
```

但是我们需要稍微修改网络。Market-1501有751个类别（不同的人），与ImageNet的1000个类别不同。所以这里我们修改了模型以使用我们的分类器（我已为你修改好了，所以你不需要修改代码。请看一下）。

```python
import torch
import torch.nn as nn
from torchvision import models

# 定义基于ResNet50的模型
class ft_net(nn.Module):
    def __init__(self, class_num = 751):   # 检查这一行。
        super(ft_net, self).__init__()
        #加载模型
        model_ft = models.resnet50(pretrained=True)
        # 将平均池化改为全局池化
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num) #定义我们的分类器。

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x) #使用我们的分类器。
        return x
```

```diff
+ Quick Question. 为什么我们使用AdaptiveAvgPool2d？AvgPool2d和AdaptiveAvgPool2d有什么区别？
+ Quick Question. 模型现在有参数吗？如何初始化新层的参数？
```
更多详情在 `model.py` 中。你可以在完成本实践后再查看。

### 第一部分3：训练（`python train.py`）
好的。现在我们准备好了训练数据并定义了模型结构。

我们可以通过以下方式训练模型：
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` 使用哪个GPU。

`--name` 模型名称。

`--data_dir` 训练数据的路径，例如 `/home/yourname/Market/pytorch`

`--train_all` 使用所有图像进行训练。

`--batchsize` 批次大小。

`--erasing_p` 随机擦除概率。

如果你遇到错误 `python 3.12+ does not support dynamic.`，我们可以（1）删除编译操作，这需要在训练和测试代码中都使用dynamic（https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/train.py#L512 和 https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/test.py#L167）。
（2）或者创建一个python 3.9的新环境
```
conda create --name python39 python=3.9
conda activate python39
```
然后重新安装所有提到的依赖。

打开另一个终端查看GPU使用情况。
```bash
nvidia-smi # 显示详细版本
pip install gpustat
gpustat # 显示简洁版本
```

现在让我们看看 `train.py` 中做什么。
首先是如何从准备好的文件夹中读取数据及其标签。
使用 `torch.utils.data.DataLoader`，我们可以获得两个迭代器 `dataloaders['train']` 和 `dataloaders['val']` 来读取数据和标签。
```python
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8) # 8个worker可能更快
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
```

这是训练模型的主要代码。
是的。只有约20行。请确保你能理解代码的每一行。
```python
            # 迭代数据。
            for data in dataloaders[phase]:
                # 获取一批输入
                inputs, labels = data
                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # 跳过最后一个batch
                    continue
                # print(inputs.shape)
                # 如果使用gpu，将数据包装成Variable并转换为cuda。
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # 将参数梯度清零
                optimizer.zero_grad()

                #-------- 前向传播 --------
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                #-------- 反向传播 + 优化 --------
                # 仅在训练阶段
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
```
```diff
+ Quick Question. 为什么我们需要optimizer.zero_grad()？如果移除它会发生什么？
+ Quick Question. 输出的维度是batchsize*751。为什么？
```
每10个训练epoch，我们保存一个快照并更新损失曲线。
```python
                if epoch%10 == 9:
                    save_network(model, epoch)
                draw_curve(epoch)
```

## 第二部分：测试
### 第二部分1：提取特征（`python test.py`）
在这一部分，我们加载网络权重（我们刚刚训练的）来提取每个图像的视觉特征。
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 60
```
`--gpu_ids` 使用哪个GPU。

`--name` 训练模型的目录名称。

`--batchsize` 批次大小。

`--which_epoch` 选择第几个模型。

`--data_dir` 测试数据的路径。

让我们看看 `test.py` 中做什么。
首先，我们需要导入模型结构，然后将权重加载到模型中。
```python
model_structure = ft_net(751)
model = load_network(model_structure)
```
对于每个查询和图库图像，我们只需通过前向传播提取特征。
```python
outputs = model(input_img)
# ---- L2归一化特征 ------
ff = outputs.data.cpu()
fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
ff = ff.div(fnorm.expand_as(ff))
```
```diff
+ Quick Question. 为什么我们在测试时水平翻转图像？如何在pytorch中fliplr？
+ Quick Question. 为什么我们对特征进行L2归一化？
```

### 第二部分2：评估
是的。现在我们有了每个图像的特征。我们需要做的唯一事情就是通过特征匹配图像。
```bash
python evaluate_gpu.py
```
如果你的结果比我们的基线差很多，比如只有10%或20%，请首先检查你的numpy版本。有人报告过numpy问题。
如果你的结果接近0或100或出错，你需要检查你的路径设置。在 `test.py` 中你设置了正确的测试路径吗？

让我们看看 `evaluate_gpu.py` 中做什么。我们对预测的相似度分数进行排序。
```python
query = qf.view(-1,1)
# print(query.shape)
score = torch.mm(gf,query) # 余弦距离
score = score.squeeze(1).cpu()
score = score.numpy()
# 预测索引
index = np.argsort(score)  # 从小到大
index = index[::-1]
```

请注意，有两类图像我们不考虑为正确匹配的图像。
* Junk_index1 是误检测图像的索引，这些图像包含身体部位。

* Junk_index2 是同一摄像机中相同身份图像的索引。

```python
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    # 不同摄像机中相同身份的图像
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # 只检测到部分身体。
    junk_index1 = np.argwhere(gl==-1)
    # 同一摄像机中相同身份的图像
    junk_index2 = np.intersect1d(query_index, camera_index)
```

我们可以使用函数 `compute_mAP` 获得最终结果。
在这个函数中，我们将忽略junk_index。
```python
CMC_tmp = compute_mAP(index, good_index, junk_index)
```

## 第三部分：简单可视化（`python demo.py`）
要可视化结果，
```
python demo.py --query_index 777
```
`--query_index` 你想测试的查询编号。你可以选择 `0 ~ 3367` 范围内的数字。

它与 `evaluate.py` 类似。我们添加了可视化部分。
```python
try: # 可视化排序结果
    # 需要图形用户界面
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(10): #显示前10张图像
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green') # 正确匹配
        else:
            ax.set_title('%d'%(i+1), color='red') # 错误匹配
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('如果你想看到排序结果的可视化，需要图形用户界面。')
```

## 第四部分：轮到你来做了。
对于作业，你可以自由选择任何相关主题。这里我只给出一些基本思路。你不需要完成所有。

- 尝试不同的数据集。Market-1501是在夏季清华大学收集的数据集。

让我们尝试另一个名为DukeMTMC-reID的数据集，它是在冬季杜克大学收集的。

你可以在[GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O)或（[百度云](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) 密码: bhbun）或使用以下bash下载数据集。自己试试吧。
```bash
gdown 1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O
python prepare_Duke.py # 请也修改路径。
```

该数据集与Market-1501非常相似。你也可以在此处查看最新结果[Here](https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/leaderboard/README-Duke.md)。

```diff
+ Quick Question. 我们能否将直接在Market-1501上训练的模型应用到DukeMTMC-reID？为什么？
```

- 尝试不同的骨干网络。 https://github.com/layumi/Person_reID_baseline_pytorch/tree/master?tab=readme-ov-file#trained-model

- 尝试不同的损失函数组合。 https://github.com/layumi/Person_reID_baseline_pytorch/tree/master?tab=readme-ov-file#different-losses

- 尝试验证+识别损失。你可以在[Here](https://github.com/layumi/Person-reID-verification)查看代码。

- 尝试Triplet Loss。
Triplet Loss是另一个广泛使用的目标。你可以在https://github.com/layumi/Person-reID-triplet-loss查看代码。
我以类似的方式编写了代码，让我们找出我改变了什么。

## 第五部分：其他相关工作
- 行人有一些特定属性，例如性别、携带物品。它们可以帮助特征学习。我们为Market-1501和DukeMTMC-reID标注了ID级属性。你可以查看[这篇论文](https://arxiv.org/abs/1703.07220)。
![](https://github.com/vana77/DukeMTMC-attribute/blob/master/sample_image.jpg)

- 我们可以使用自然语言作为查询吗？查看[这篇论文](https://arxiv.org/abs/1711.05535)。
![](https://github.com/layumi/Image-Text-Embedding/blob/master/CUHK-show.jpg)

- 我们可以使用其他损失函数（即对比损失）来进一步提高性能吗？查看[这篇论文](https://arxiv.org/abs/1611.05666)。
![](https://github.com/layumi/2016_person_re-ID/raw/master/paper.jpg)

- 人员重识别数据集不够大来训练深度学习网络？你可以查看[这篇论文](https://arxiv.org/abs/1701.07717)（使用GAN生成更多样本）并尝试一些数据增强方法，如[随机擦除](https://arxiv.org/abs/1708.04896)。

![](https://github.com/layumi/Person-reID_GAN/raw/master/fig0.jpg)

- 行人检测效果不好？尝试[Open Pose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)和[Spatial Transformer](https://github.com/layumi/Pedestrian_Alignment)来对齐图像。

![](https://github.com/layumi/Pedestrian_Alignment/raw/master/fig2.jpg)

- 数据有限？生成更多！[Code](https://github.com/NVlabs/DG-Net)

![](https://github.com/NVlabs/DG-Net/raw/master/NxN.jpg)

- 3D人员重识别 [Code](https://github.com/layumi/person-reid-3d)
![](https://github.com/layumi/person-reid-3d/blob/master/imgs/demo-1.jpg)

## 快速问题答案
你可以查看 https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/tutorial/Answers_to_Quick_Questions.md

## Star历史

如果你喜欢这个仓库，请star它。非常感谢！

<img width="884" alt="image" src="https://github.com/user-attachments/assets/c16e8341-8a78-4e62-addc-9af610690e65" />


[![Star History Chart](https://api.star-history.com/svg?repos=layumi/Person_reID_baseline_pytorch&type=Date)](https://star-history.com/#layumi/Person_reID_baseline_pytorch&Date)

## 参考文献

[1] Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "Imagenet: A large-scale hierarchical image database." In Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on, pp. 248-255. Ieee, 2009.

[2] Zheng, Zhedong, Liang Zheng, and Yi Yang. "Unlabeled samples generated by gan improve the person re-identification baseline in vitro." In Proceedings of the IEEE International Conference on Computer Vision, pp. 3754-3762. 2017.

[3] Zheng, Zhedong, Xiaodong Yang, Zhiding Yu, Liang Zheng, Yi Yang, and Jan Kautz. "Joint discriminative and generative learning for person re-identification." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2138-2147. 2019.

[4] Zheng, Zhedong, Liang Zheng, and Yi Yang. "A discriminatively learned cnn embedding for person reidentification." ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) 14, no. 1 (2017): 1-20.

[5] Zheng, Zhedong, Liang Zheng, and Yi Yang. "Pedestrian alignment network for large-scale person re-identification." IEEE Transactions on Circuits and Systems for Video Technology 29, no. 10 (2018): 3037-3045.

[6] Zheng, Zhedong, Liang Zheng, Michael Garrett, Yi Yang, and Yi-Dong Shen. "Dual-path convolutional image-text embedding with instance loss." ACM TOMM 2020.

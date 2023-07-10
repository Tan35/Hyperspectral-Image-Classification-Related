import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import time 
import os


parser = argparse.ArgumentParser(description='HybridSN-DDP-Training')
parser.add_argument('--local_rank',
                    default=-1,
                    type=int,
                    help='node rank for distributed training')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')


class HybridSN(nn.Module):
    def __init__(self, num_classes=16):
        super(HybridSN, self).__init__()
        # 第一层三维卷积：Conv3d_1, Input:(1, 30, 25, 25), 8 个 7 × 3 × 3 的卷积核
        # Output: (8, 24, 23, 23)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True) # 使用 inplace=True 的好处是不用开辟新的内存空间
        )

        # 第二层三维卷积：Conv3d_2，Input:(8, 24, 23, 23)，16 个 5 × 3 × 3 的卷积核
        # Output: (16, 20, 21, 21)
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # 第三层三维卷积：Conv3d_3, Input(16, 20, 21, 21)，32 个 3 × 3 × 3 的卷积核
        # Output: (32, 18, 19, 19)
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # 第四层是二维卷积，但是要二维卷积之前就要把第三层的输出 reshape 成可以二维卷积的形状
        # 其实就是样本数和维度数堆叠：32 × 18 = 576，所以这一层二维卷积的输入为 576
        # Output: (64, 17, 17)
        self.conv4_2d = nn.Sequential(
            nn.Conv2d(576, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # 二维卷积层之后就是一个全连接层，而在送入这个全连接层之前，要把二维卷积之后的输出降为一维才能送入全连接层
        # 所以在 forward 里直接使用 flatten()，那么输入全连接层参数即为：64 × 17 × 17 = 18496
        # 这一全连接层有 256 个节点
        self.dense1 = nn.Linear(18496, 256)

        # 接着又是一个全连接层，有 128 个节点
        self.dense2 = nn.Linear(256, 128)

        # 最终输出层，有 num_classes 这么多节点，因为我们最后就是要分 num_classes 这么多地物类别（Indian Pines）数据集
        self.dense_out = nn.Linear(128, num_classes)
        
        
        # 论文中使用了 Dropout ，参数为 0.4
        self.dropout = nn.Dropout(0.4)

    # 定义前向传播函数
    def forward(self, x):
        out = self.conv1(x)  # 第一层三维卷积
        out = self.conv2(out)  # 第二层三维卷积
        out = self.conv3(out)  # 第三层三维卷积 # 32, 18, 19, 19

        out = self.conv4_2d(out.reshape(out.shape[0], -1, 19, 19))  # 第四层二维卷积
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.dropout(self.dense1(out)))
        out = F.relu(self.dropout(self.dense2(out)))
        out = self.dense_out(out)
        return out
    

# 对高光谱数据集 X 应用 PCA 变换
def applyPCA(X, num_components):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
# 经过 padWithZeros 之后，宽、高会变成 169,169， 也就是左右各 padding 了 12
def padWithZeros(X, margin=12):
    # print(X)
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    # print(newX.shape)  # (169, 169, 30) 全是 0
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

#  对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)  # 12
    zeroPaddedX = padWithZeros(X, margin=margin)  # (169, 30, 30)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    # print(patchesData.shape) # (21025, 25, 25, 200)
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    # print(patchesLabels.shape) # (21025, )
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):  # for r in range(12, 169-12)
        for c in range(margin, zeroPaddedX.shape[1] - margin):  # for c in range(12, 169-12)
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            # print(patch.shape)
            # zeroPaddedX[r - 12 : r + 12 + 1, c -12 : c + 12 + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    # 如果没有 removeZeroLabels
    # patchesData 的 shape 将会是 (21025, 25, 25, 30)
    # patchesLabels 的 shape 将会是 (21025, )
    # removeZeroLabels 的作用就是去掉 gt 标签集 groundtruth 中为 0 的数据，因为里面的数据值有 0~16，而刚好 1~16 刚好对应地物分类的 16 类
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


# 随机划分训练集和测试集
# 这里最后取 testradio=0.9 也就是随机抽 90% 为测试集，10% 为训练集
def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test


# 读取数据集（使用 Indian Pines 数据集）
X = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

# 用于测试样本的比例
test_ratio = 0.90
# 每个像素周围提取 patch 的尺寸
patch_size = 25
# 使用 PCA 降维，得到主成分的数量
pca_components = 30

print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, num_components=pca_components)
print('Data shape after PCA: ', X_pca.shape)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
Xtest  = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
print('before transpose: Xtrain shape: ', Xtrain.shape) 
print('before transpose: Xtest  shape: ', Xtest.shape) 


# 为了适应 pytorch 结构，数据要做 transpose
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
Xtest  = Xtest.transpose(0, 4, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape) 
print('after transpose: Xtest  shape: ', Xtest.shape) 


""" Training dataset"""
class TrainDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)        
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset): 
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self): 
        # 返回文件数据的数目
        return self.len

# 这里暂时先不创建 创建 trainloader 和 testloader，因为使用 DDP 要有单独的方法
# 创建 trainloader 和 testloader
# trainset = TrainDS()
# testset  = TestDS()
# train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
# test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=128, shuffle=False, num_workers=2)



# 查看 cuda 是否可用，可用情况下打印可用 GPU(s) 相关信息
if torch.cuda.is_available():
    gpus = []
    print(f"Found {torch.cuda.device_count()} GPUs available ✔")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        gpus.append(i)
        # nn.DataParallel 多卡汇总，用于汇总多个GPU的计算结果并输出模型的训练过程和性能指标，通常拿第一张卡
        output_device = gpus[0]
    print(f'GPUS_ID: {gpus}')
    print(f'OUTPUT_GPU_ID: {output_device}')
else:
    print("No GPUs found")
    

def main():
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))
    

def main_worker(local_rank, nprocs, args):
    
    args.local_rank = local_rank
    
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=args.nprocs,
                            rank=local_rank)
    
    torch.cuda.set_device(local_rank)
    
    model = HybridSN()
    model.cuda(local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    
    args.batch_size = int(args.batch_size / nprocs)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.Adam(model.parameters(), lr =0.001)
    
    # 这里开始创建 DDP 的 trainloader 和 testloader
    trainset = TrainDS()
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    # testset  = TestDS()
    # test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    # test_loader = torch.utils.data.DataLoader(testset,
    #                                          batch_size=args.batch_size,
    #                                          num_workers=2,
    #                                          pin_memory=True,
    #                                          sampler=test_loader)
    
    for epoch in range(100):

        train_sampler.set_epoch(epoch)
        # val_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, local_rank, args)


def train(train_loader, model, criterion, optimizer, epoch, local_rank, args):
    
    # 开始训练
    model.train()
    s_time = time.time()
    total_loss = 0
    total_step = 100
    for epoch in range(100):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda(local_rank)
            labels = labels.cuda(local_rank)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 正向传播 +　反向传播 + 优化 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))
    e_time = time.time()
    train_time = e_time - s_time
    print(f"Training time: {train_time}")
    print('Finished Training')
    # 判断是否使用了多 GPU 进行训练
    if torch.cuda.device_count() > 1 and len(model.device_ids) > 1:
        print('This code is trained on multiple GPUs.')
    else:
        print('This code is not trained on multiple GPUs.')
    
if __name__ == '__main__':
    main()
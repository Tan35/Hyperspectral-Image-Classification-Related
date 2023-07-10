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
import time
import argparse
import os
import shutil


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

parser = argparse.ArgumentParser("HybridSN-DISTRIBUTED-DP-TRAINING")

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--model', type=str, default='HybridSN', help='select network to train')


args = parser.parse_args()
print(args)
# args.save = 'IN-train-model-{}-DP-{}-lr{}'.format(args.model, time.strftime("%Y%m%d-%H%M%S"), args.learning_rate)
args.save = 'IN-{}-DP-{}-lr-{}-batch_size-{}'.format(args.model, time.strftime("%Y%m%d-%H%M%S"), args.learning_rate, args.batch_size)
create_exp_dir(args.save, scripts_to_save=['DP-HybridSN.py'])
SAVE_PATH = args.save + '/' + 'HybridSN'+ '_sample' + '.pth' 


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
X = sio.loadmat('Indian_pines_corrected')['indian_pines_corrected']
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


# 可用 GPU 个数
num_gpu = torch.cuda.device_count()

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset  = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128 * num_gpu, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(dataset=testset,  batch_size=128 * num_gpu, shuffle=False, num_workers=2)


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


model = HybridSN()
model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 开始训练
s_time = time.time()
total_loss = 0
total_step = 100
for epoch in range(100):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' %(epoch + 1, total_loss/(epoch+1), loss.item()))
torch.save(model.state_dict(), SAVE_PATH)
e_time = time.time()
train_time = e_time - s_time
print(f"Training time: {train_time}")
print('Finished Training')


count = 0
model = model.eval()
# 模型测试
for inputs, _ in test_loader:
    inputs = inputs.cuda()
    outputs = model(inputs)
    outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
    if count == 0:
        y_pred_test =  outputs
        count = 1
    else:
        y_pred_test = np.concatenate( (y_pred_test, outputs) )

# 生成分类报告
classification = classification_report(ytest, y_pred_test, digits=4)
print(classification)
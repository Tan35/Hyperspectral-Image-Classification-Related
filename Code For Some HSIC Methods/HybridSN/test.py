import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# --- 定义 HybridSN 类 ---
# 3 层 三维卷积，一层二维卷积，三层全连接层


class HybridSN(nn.Module):
    def __init__(self, num_classes=16):
        super(HybridSN, self).__init__()
        # 第一层三维卷积：Conv3d_1, Input:(1, 30, 25, 25), 8 个 7 × 3 × 3 的卷积核
        # Output: (8, 24, 23, 23)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), stride=1, padding=0),
            nn.ReLU()
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
        out = self.conv3(out)  # 第三层三维卷积
        out = self.conv4_2d(out.reshape(out.shape[0], -1, 19, 19))  # 第四层二维卷积
        out = out.reshape(out.shape[0], -1)
        out = F.relu(self.dropout(self.dense1(out)))
        out = F.relu(self.dropout(self.dense2(out)))
        out = self.dense_out(out)
        return out


# 对高光谱数据集 X 应用 PCA 变换
def applyPCA(X, num_components):
    newX = np.reshape(X, (-1, X.shape[2]))
    # print(type(newX))
    # print(newX.shape) # (145*145=21025, 200)
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    # print(newX.shape) # (21025, 30)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    # print(newX.shape)
    # print("------")
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


# 构建训练数据集
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


# 构建测试数据集
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


# 读取 Indian Pines 数据集
X = sio.loadmat('Indian_pines_corrected.mat')['indian_pines_corrected']
y = sio.loadmat('Indian_pines_gt.mat')['indian_pines_gt']

# PCA 降维后的通道数大小
num_components = 30

# 每个像素周围提取 patch 的尺寸
patch_size = 25

# 用于测试样本的比例
test_ratio = 0.90

pca_X = applyPCA(X, num_components)  # (145, 145, 30)
print(f' --- Apply X to PCA, shape is {pca_X.shape} ---')
# padWithZeros_X = padWithZeros(pca_X,margin=12) # 边缘像素已经填充 0，原有数据不变，现在的 shape 是 (169, 169, 30)
X_patches, y_patches = createImageCubes(pca_X, y, windowSize=patch_size)
print(" --- already split some small cubes √√√--- ")
print(f'the small X cubes shape is {X_patches.shape}, the small y cubes shape is {y_patches.shape}')

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_patches, y_patches, test_ratio)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

# 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, num_components, 1)
Xtest = Xtest.reshape(-1, patch_size, patch_size, num_components, 1)
print('before transpose: Xtrain shape: ', Xtrain.shape)
print('before transpose: Xtest  shape: ', Xtest.shape)

# 为了适应 pytorch 结构，数据要做 transpose
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
Xtest = Xtest.transpose(0, 4, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape)
print('after transpose: Xtest  shape: ', Xtest.shape)

# 创建 trainloader 和 testloader
trainset = TrainDS()
testset = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False, num_workers=2)

# 选择 GPU 进行训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
net = HybridSN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 开始训练
total_loss = 0
for epoch in range(100):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 +　反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1, total_loss / (epoch + 1), loss.item()))

print('Finished Training')

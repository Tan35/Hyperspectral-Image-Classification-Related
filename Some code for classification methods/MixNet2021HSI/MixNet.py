import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from torch.autograd import Variable

# 这个模块为MLB-A模块
# 没有做到完全的格式自动适配
class MLB_A_Inception(nn.Module):
    # c1 - c4为每条线路⾥的层的输出通道数
    def __init__(self,inplanes = 48*2, k=48, k2=12, dropRate=0.4):
        super(MLB_A_Inception,self).__init__()

        # 可以用于复用的卷积模块操作。
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
        nn.ReLU(inplace=True),
        nn.Conv2d(inplanes, (inplanes + k) // 2, kernel_size=1),
        nn.Dropout(p=dropRate),
        nn.BatchNorm2d((inplanes + k) // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d((inplanes + k) // 2, k, kernel_size=3, padding=1),
        nn.Dropout(p=dropRate),
        )

    def forward(self,x):
        x_UseRes = self.conv1(x)
        x_UseDense = self.conv1(x)
        # inputX : [batchSize,x_Inplanes,inputHeigh,inputWidth]
        # eg:[128,96,13,13]
        # 这里定好了数据的输入格式，因此无法更好得优化代码
        # eg:[128,48:96,13,13]
        x_shortcut = x[:,x.shape[1] - x_UseRes.shape[1]:x.shape[1],:,:]
        x_shortcut = x_UseRes + x_shortcut
        x_unprocess = x[:,:x.shape[1] - x_UseRes.shape[1],:,:]
        # 直接将这么将通道进行稠密连接的操作。
        # x_unprocess不进行处理层，x_shortcut残差连接层，x_UseDense稠密连接层
        return torch.cat((x_unprocess,x_shortcut,x_UseDense),dim=1) # 在通道维上连结输出

# 这个模块为MLB-B模块
class MLB_B_Inception(nn.Module):
    # c1 - c4为每条线路⾥的层的输出通道数
    def __init__(self,inplanes = 48*2, k=48, k2=12, dropRate=0.4):
        super(MLB_B_Inception,self).__init__()

        # 可以用于复用的卷积模块操作。
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, (inplanes + k) // 2, kernel_size=1),
            nn.Dropout(p=dropRate),
            nn.BatchNorm2d((inplanes + k) // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d((inplanes + k) // 2, k, kernel_size=3, padding=1),
            nn.Dropout(p=dropRate),

        )

        # # 线路2，卷积等一系列操作后用于稠密连接
        # self.bn2_1 = nn.BatchNorm2d(inplanes)
        # self.relu2_1 = nn.ReLU(inplace=True)
        # self.conv2_1 = nn.Conv2d(inplanes, (inplanes + k)//2, kernel_size=1, bias=False)
        # self.bn2_2 = nn.BatchNorm2d((inplanes + k)//2)
        # self.relu2_2 = nn.ReLU(inplace=True)
        # self.conv2_2 = nn.Conv2d((inplanes + k)//2, k, kernel_size=3, padding=1, bias=False)

    def forward(self,x):
        x_UseRes = self.conv1(x)
        x_UseDense = self.conv1(x)
        # x_shortcut = x[len(x_UseDense):len(x)]
        x_append = x_UseRes + x_UseDense

        # 直接将这么将通道进行稠密连接的操作。
        # return torch.cat((p1,p2,p3,p4),dim=1) # 在通道维上连结输出
        return torch.cat((x,x_append),dim=1) # 在通道维上连结输出

# 改类设置成最后一步的操作函数
class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        out = F.avg_pool2d(out, 2)
        return out

class MixNet(nn.Module):

    # 初始化的时候，
    # unit代表复用哪个模块,
    # unitTimes = 3,服用改模块额次数
    # inplanes代表了输入图像的层数
    def __init__(self,
                 inplanes = 1,
                 InputWidth = 2,
                 InputHeigh = 2,
                 unit=MLB_B_Inception,
                 unitTimes = 3,
                 dropRate= 0.4,
                 num_classes=10,
                 k=48,
                 compressionRate=2):
        super(MixNet, self).__init__()

        self.k = k
        self.dropRate = dropRate
        self.inplanes = inplanes
        # 原论文，先将输入图像的通道转为2k
        self.conv1 = nn.Conv2d(inplanes, 2 * k, kernel_size=3, padding=1)
        # 这里的block主要调用三个MLB的模块，从2k到5k通道的一个变化，由于步骤重复，就不进行冗余写法
        # 重复多少次，unit = 3次，自由计算最后的结果
        self.block1 = self._make_block(unit, 2 * k,unitTimes)

        # 最后一步的卷积操作
        # self.trans1 = self._make_transition(compressionRate)

        self.bn = nn.BatchNorm2d(2 * k + self.k * unitTimes)
        self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(8)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # self.fc = nn.Linear(self.inplanes, num_classes)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=(2 * k + self.k * unitTimes)* (InputWidth//2) * (InputHeigh//2),
                out_features=num_classes
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=(2 * k + self.k * unitTimes)* (InputWidth//2) * (InputHeigh//2),
                out_features=256
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=256,
                out_features=128
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=128,
                out_features=num_classes
            )
        )

        self.dropout = nn.Dropout(p=self.dropRate)

    def _make_block(self, unit, BlockInplanes,unitTimes):
        layers = []
        BlockInplanes = BlockInplanes
        for i in range(unitTimes):

            # Currently we fix the expansion ratio as the default value
            layers.append(unit(BlockInplanes, k=self.k, dropRate=self.dropRate))
            # 每次都将blockInplanes 的最新的输出层数进行更新
            BlockInplanes += self.k

        return nn.Sequential(*layers)

    def forward(self, x):
        # 数据预处理
        x = self.conv1(x)
        # 加入一个去除一些网络连接，降低整个模型复杂连接
        x = self.dropout(x)
        # 服用MLB结构
        x = self.block1(x)
        # 进行常规操作输出整个结果
        x = self.bn(x)
        # 加入一个去除一些网络连接，降低整个模型复杂连接
        x = self.dropout(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = self.classifier(x)

        return x

def mixnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return MixNet(**kwargs)

if __name__ == '__main__':
    # print(MLB_B_Inception())
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    # 网络放到GPU上
    # net = MixNet(num_classes=16).to(device)
    print(MixNet(unit = MLB_A_Inception))

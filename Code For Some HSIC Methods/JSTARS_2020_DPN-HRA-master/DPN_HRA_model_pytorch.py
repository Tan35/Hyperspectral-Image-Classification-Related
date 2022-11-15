import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# 这个可以做下测试，进行一定的调整，但总体来说不是很复杂
class RBAM_Attention(nn.Module):
    def __init__(self,nout=1,reduce=1,nin = 1):
        super(RBAM_Attention, self).__init__()

        self.reduce = reduce
        self.gp = nn.AvgPool2d(1)
        self.relu = nn.LeakyReLU(inplace=True)
        # 论文中的全连接层的写法有点问题，这里并不是真正的全连接层
        # self.se = nn.Sequential(
        #     nn.Linear(nout,nout // reduce),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(nout // reduce,nout),
        #     nn.Sigmoid()
        # )
        self.se = nn.Sequential(
            nn.Conv2d(in_channels = nout,
                      out_channels= nout // reduce,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=nout // reduce,
                      out_channels=nout,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.Sigmoid()
        )

    def forward(self,x):
        # 参照论文的要求写法来编写
        # 有关参数调整慢慢修改
        input = x.clone()
        BatchSize,C,D,H,W = x.shape
        output = x.view(BatchSize,C*D,H,W)
        # output_size:10,1000,1,1?
        # output = self.gp(output)
        output = F.avg_pool2d(output,(H,W))
        output = output.view(BatchSize,C,D,1)
        output = self.se(output)

        # 不扩展，直接进行点乘即可
        for i in range(BatchSize):
            for j in range(C):
                for k in range(D):
                    input[i,j,k,:,:] = torch.mul(x[i,j,k,:,:],output[i,j,k])
        # 点乘
        return input

# 这个模块为RBAM模块,类似与SE-Net的部分设计
class RBAM_Inception(nn.Module):
    # c1 - c4为每条线路⾥的层的输出通道数
    def __init__(self,inplanes,reduce = 1):
        super(RBAM_Inception,self).__init__()
        # 可以用于复用的卷积模块操作。
        C = inplanes
        # 作者在设计的时候保持了填充的操作
        self.conv3D_1 = nn.Sequential(
            nn.Conv3d(in_channels = C,
                      out_channels= C,
                      kernel_size=(5, 3, 3),
                      stride=1,
                      padding=(2,1,1)),
            nn.BatchNorm3d(C),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=C,
                      out_channels=C,
                      kernel_size=(5, 3, 3),
                      stride=1,
                      padding=(2,1,1)),
        )
        # 这一块主要是SE-Net中的操作
        # 在SENet中的挤压通道的参数
        self.reduce = reduce
        self.RBAM_Attention = RBAM_Attention(nout = C,reduce = 1)
    def forward(self,x):
        # 首先一顿3D卷积操作
        input = x.clone()
        x_3Dconv1 = self.conv3D_1(input)
        # 接着一堆SENet的操作
        x_SE_3D_Attention = self.RBAM_Attention(x_3Dconv1)
        # 将原先的输入特征feature_map与经过一系列处理的结果进行相加
        # 这里一部分是类似于残差相加的结构，另一部分是SE挤压通道额操作
        out = x + x_SE_3D_Attention
        return out

# 这个可以坐下测试，进行一定的调整，但总体来说不是很复杂
class RSAM_Attention(nn.Module):
    def __init__(self,nin=1,nout=1,reduce=1):
        super(RSAM_Attention, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.se_RSAM = nn.Sequential(
            nn.Conv2d(in_channels = nout,
                      out_channels= 1,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.Sigmoid(),
        )

    def forward(self,x):
        # 参照论文的要求写法来编写
        # 有关参数调整慢慢修改
        input = x.clone()
        output = self.se_RSAM(input)
        # 进行注意力卷积的乘积的操作

        # for i in range(input.shape[0]):
        #     for j in range(input.shape[1]):
        #         for k i
        #         # x.mul(y)
        #         # input[i, j, :, :] = input[i, j, :, :] * output[i, 0, :, :]
        #         input[i, j, :, :] = torch.mul(input[i, j, :, :],output[i, 0, :, :])
        # for i in range(input.shape[0]):
        #     for j in range(input.shape[1]):
        #         for k in range(input.shape[2]):
        #             for l in range(input.shape[3]):
        #                 input[i, j, k, l] = input[i, j, k, l]*output[i, 0, k, l]
        # input = input * output[:, 0, :, :]
        for j in range(input.shape[1]):
            input[:,j,:,:] = torch.mul(x[:,j,:,:], output[:, 0, :, :])

        return input

# RSAM的网络操作类似，只不过是2D维度进行操作

class RSAM_Inception(nn.Module):
    # c1 - c4为每条线路⾥的层的输出通道数
    def __init__(self,inplanes = 1,reduce = 1):
        super(RSAM_Inception,self).__init__()
        C = inplanes
        # 可以用于复用的卷积模块操作。
        self.conv2D_1 = nn.Sequential(
            nn.Conv2d(in_channels = C,
                      out_channels= C,
                      kernel_size=( 3, 3),
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(C),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=C,
                      out_channels=C,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
        )
        # 这一块主要是SE-Net中的操作
        self.RSAM_Attention = RSAM_Attention(nout=C,reduce=1)

    def forward(self,x):
        # 首先一顿3D卷积操作
        x_2Dconv1 = self.conv2D_1(x)
        # 接着一堆SENet的操作
        x_SE_2D_Attention = self.RSAM_Attention(x_2Dconv1)
        # 将原先的输入特征feature_map与经过一系列处理的结果进行相加
        out = x + x_SE_2D_Attention
        # 这里一部分是类似于残差相加的结构，另一部分是SE挤压通道额操作

        return out
# 大部分代码差别估计就在

class DPN_HRA(nn.Module):
    def __init__(self, num_classes,BatchSize=1,C=1,D=1,H=11,W=11):
        super(DPN_HRA, self).__init__()
        self.BatchSize = BatchSize
        self.C = C
        self.D = D
        self.H = H
        self.W = W
        # 3D处理模块
        # 3D卷积块
        self.block_1_3D = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=64,
                kernel_size=(7, 3, 3),
                stride=1,
                padding=(3,1,1)
            ),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
        )
        # RBAM网络,这个reduce可以自由设置
        reduce = 2
        self.RBAM_Inception = RBAM_Inception(inplanes=64,reduce=reduce)
        # 2D处理模块
        self.block_2_1BN_ReLU = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True),
        )
        # 一个reshape的操作，在forward写
        # 一个卷积操作，在forward写，同时光谱维度的层数为128
        self.block_2_2Conv = nn.Conv2d(
                in_channels=D*64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=1,
                padding=1
            )

        self.block_2_3BN_ReLU = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.RSAM_Inception = RSAM_Inception(inplanes=128,reduce=1)
        # 1D处理模块
        # 这个数字需要输入计算
        self.block_3_1D = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )
        self.block_3_1D_Avg = nn.AvgPool2d(
            (H,W),
            stride=1,
        )
        # 直接进行fc层的一个操作
        self.fc_1d = nn.Sequential(
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=64,
                out_features=num_classes
            ),

            # pytorch交叉熵损失函数是混合了softmax的。不需要再使用softmax
        )

    def forward(self,x):
        # 3D模块处理
        output_3D = self.block_1_3D(x)
        # RBAM的C为第四维度的参数传入
        # C = x.shape[3]
        # output_3D = RBAM_Inception(output_3D,inplanes = C)
        output_3D = self.RBAM_Inception(output_3D)

        # 2D模块处理
        output_2D =self.block_2_1BN_ReLU(output_3D)
        # reshape操作
        output_2D = output_2D.view(output_2D.shape[0],
                                   output_2D.shape[1] * output_2D.shape[2] ,
                                   output_2D.shape[3],
                                   output_2D.shape[4])
        # # 128种，对应feature map的size的3*3卷积
        # w = torch.rand(128,output_2D.shape[1],3,3)
        # # 和卷积核种类保持一致（不同通道共用一个bias）
        # b = torch.rand(128)
        # # 需要将设备的选择保持一致的格式
        # w = w.to(device)
        # b = b.to(device)
        # # 步长为1，不进行0填充
        # output_2D = F.conv2d(output_2D,w,b,stride=1,padding=0)
        # 替换写法，不得已之举
        output_2D = self.block_2_2Conv(output_2D)
        # 2D模块处理
        output_2D = self.block_2_3BN_ReLU(output_2D)
        # RSAM模块处理
        output_2D = self.RSAM_Inception(output_2D)

        # 1D模块处理
        output_1D = self.block_3_1D(output_2D)

        # BatchSize, C,  H, W = output_1D.shape
        # output_1D  = F.avg_pool2d(output_1D, (H, W))

        output_1D = self.block_3_1D_Avg(output_1D)

        output_1D = output_1D.view(self.BatchSize,-1)
        finaloutput = self.fc_1d(output_1D)


        return finaloutput

if __name__ == '__main__':
    # num_classes = 16
    # print(DPN_HRA(num_classes))
    # ok，3D部分的代码已经设置好了
    # testInput = torch.ones((10,64,10,11,11))
    # net = RBAM_Inception(inplanes=64,reduce=2)
    # print(net(testInput))

    # 测试2D部分的结构代码
    # testInput = torch.ones((10,10,10,10))
    # net = RSAM_Inception(inplanes=10,reduce=1)
    # print(net(testInput))

    # 全部测试完毕，组合下网络即可

    testInput = torch.ones((10,1,10,11,11))
    BatchSize, C,D, H, W = testInput.shape
    num_classes = 16
    net = DPN_HRA(num_classes,
                  BatchSize = BatchSize,
                  C = C,
                  D = D,
                  H = H,
                  W = W)
    print(net)
    print(net(testInput))

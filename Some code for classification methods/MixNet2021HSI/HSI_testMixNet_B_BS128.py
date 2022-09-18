import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import spectral
from operator import truediv
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import time
import cv2
from MixNet import *

def loadDatasSet(loadDataSet = 'indian_pine'):
    # 判断选择的数据集的类型
    if loadDataSet == 'indian_pine':
        # Indian Pines美国印第安纳州一块印度松树
        X = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']

    elif loadDataSet == 'PaviaU':
        # Pavia University
        X = sio.loadmat('../data/PaviaU.mat')['paviaU']
        y = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt']

    elif loadDataSet == 'salinas_corrected':
        # 加州 Salinas Valley
        X = sio.loadmat('../data/Salinas_corrected.mat')['salinas_corrected']
        y = sio.loadmat('../data/Salinas_gt.mat')['salinas_gt']

    elif loadDataSet == 'PHI_FangluTeaFarm':
        # 方麓茶场
        X = sio.loadmat('../data/PHI_FangluTeaFarm.mat')['PHI_FangluTeaFarm']
        y = sio.loadmat('../data/PHI_GroundTruthFanglu.mat')['PHI_GroundTruthFanglu']
    elif loadDataSet == 'KSC':
        # 方麓茶场
        X = sio.loadmat('../data/KSC.mat')['KSC']
        y = sio.loadmat('../data/KSC_gt.mat')['KSC_gt']
    else:
        # 防止异常情况的发送，选用indian作为保底
        X = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
        y = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']

    return X,y

# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    # 加入零边界，并且，将原先的X赋值进入到新创建的newX中，假设尺寸为145*145，边界大小选择左右填充12和12，则新的图像为169（145+24） * 169
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    # 给 X 做 padding填充0像素
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    # 对于每一个像素都取出其边界的所有的点，并且放入patchesData这个数组中，并且将y标注是什么内容都放入其中进行标记
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    # 移除0的标签，即没有使用到的图像标注
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式,没有将0像素点取消掉
def createImageCubesNoDeleteZero(X, y, windowSize=5, removeZeroLabels = True):
    # 给 X 做 padding填充0像素
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    # 对于每一个像素都取出其边界的所有的点，并且放入patchesData这个数组中，并且将y标注是什么内容都放入其中进行标记
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    # 移除0的标签，即没有使用到的图像标注
    # if removeZeroLabels:
    #     patchesData = patchesData[patchesLabels>0,:,:,:]
    #     patchesLabels = patchesLabels[patchesLabels>0]
    #     patchesLabels -= 1
    return patchesData, patchesLabels



def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test

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

""" Testing dataset """
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

def sklearnSaveReport(OrientLabel,PredictLabel):
    # 文件名看是否有指定，参数没有指定的话会自己创建一个

    # 这一步
    # classification = classification_report(OrientLabel, PredictLabel )
    # # 打印出来分类的结果
    # print(classification)
    # 生成分类报告

    classification, confusion,  oa, each_acc, aa, kappa = reports(OrientLabel, PredictLabel)
    classification = str(classification)
    confusion = str(confusion)
    # 保存好运行的数据
    file_name = str((time.asctime(time.localtime(time.time())))) + 'report' + '.txt'
    # file_name =  'report1' + '.txt'
    loadpath = ''
    file_name = loadpath + file_name
    with open(file_name, 'w') as x_file:
        # 不是神经网络，没有打的损失值输出
        # x_file.write('{} Test loss (%)'.format(Test_loss))
        # x_file.write('\n')
        # x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
        # x_file.write('\n')
        x_file.write('{} data (%)'.format(str((time.asctime(time.localtime(time.time()))))))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write(confusion)
        x_file.write('\n')
# 预测数据
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)                        #获取confusion_matrix的主对角线所有数值
    list_raw_sum = np.sum(confusion_matrix, axis=1)              #将主对角线所有数求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))   #list_diag/list_raw_sum  对角线各个数字/对角线所有数字的总和
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports(OrientLabel,PredictLabel):
    # start = time.time()

    # end = time.time()
    # print(end - start)

    classification = classification_report(OrientLabel, PredictLabel, digits = 5)
    oa = accuracy_score(OrientLabel, PredictLabel)  # 计算OA
    confusion = confusion_matrix(OrientLabel, PredictLabel)  # 计算confusion
    each_acc, aa = AA_andEachClassAccuracy(confusion)  # 计算each_acc和aa
    kappa = cohen_kappa_score(OrientLabel, PredictLabel)  # 计算kappa
    # score = model.evaluate(x_test, y_test, batch_size=32)
    # Test_Loss = score[0] * 100
    # Test_accuracy = score[1] * 100

    return classification, confusion,  oa * 100, each_acc * 100, aa * 100, kappa * 100



if __name__ == '__main__':


    loadDataSet = 'indian_pine'
    # loadDataSet = 'PaviaU'
    # loadDataSet = 'salinas_corrected'
    # loadDataSet = 'PHI_FangluTeaFarm'
    # loadDataSet = 'KSC'

    X,y = loadDatasSet(loadDataSet = loadDataSet)

    # 地物类别
    class_num = y.max()

    # plt.imshow(y.astype(int))
    # plt.show()
    # 用于测试样本的比例
    test_ratio = 0.90
    # 每个像素周围提取 patch 的尺寸
    patch_size = 25
    # 使用 PCA 降维，得到主成分的数量
    # 这里替换成数据集所原有的大小
    pca_components = X.shape[2]

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    # X_pca = X
    print('\n... ... PCA tranformation ... ...')
    pca_components = 30
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    # 划分训练集和测试集，test_ratio = 0.90
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose，这里的作用是调整数据的摆放位置
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain[:,0,:,:,:]
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest[:,0,:,:,:]
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建 trainloader 和 testloader 将数据放入其中
    trainset = TrainDS()
    testset = TestDS()



    train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False, num_workers=2)

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 网络放到GPU上
    net = MixNet(inplanes= pca_components,unit = MLB_B_Inception,InputHeigh=patch_size,InputWidth=patch_size,num_classes = class_num).to(device)
    # 选择交叉熵函数
    criterion = nn.CrossEntropyLoss()
    # 选择adam优化器，学习率为0.001
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 开始训练
    # 训练100轮
    total_loss = 0
    for epoch in range(30):
        for i, (inputs, labels) in enumerate(train_loader):
            # 把数据送到设备中，如果有cuda放入cuda中
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
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (
        epoch + 1, total_loss / (epoch + 1), loss.item()))

    print('Finished Training')

    count = 0
    # 模型测试
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))

    # 生成分类报告
    classification = classification_report(ytest, y_pred_test, digits=4)
    sklearnSaveReport(ytest, y_pred_test)

    print(classification)
    # load the original image
    # X = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
    # y = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']

    X,y = loadDatasSet(loadDataSet = loadDataSet)
    height = y.shape[0]
    width = y.shape[1]

    X = applyPCA(X, numComponents=pca_components)
    X = padWithZeros(X, patch_size // 2)

    # 逐像素预测类别

    outputs = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if int(y[i, j]) == 0:
                continue
            else:
                image_patch = X[i:i + patch_size, j:j +patch_size, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                  1)
                X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                X_test_image = X_test_image[:,0,:,:,:]
                prediction = net(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')
    np.save('resultNpy/saveoutput',outputs)
    cv2.imwrite("figure/finalpre" +  ".jpg", outputs)

    # X, y = loadDatasSet(loadDataSet=loadDataSet)
    # height = y.shape[0]
    # width = y.shape[1]
    #
    # X = applyPCA(X, numComponents=pca_components)
    # X = padWithZeros(X, patch_size // 2)
    # count = 0
    # # 逐像素预测类别
    # countIndex = 0
    # outputs = np.zeros((height, width))
    # for i in range(height):
    #     for j in range(width):
    #         if int(y[i, j]) == 0:
    #             continue
    #         else:
    #             Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    #             print('Xtrain shape: ', Xtrain.shape)
    #             print('Xtest  shape: ', Xtest.shape)
    #
    #             # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    #             Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    #             Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    #             print('before transpose: Xtrain shape: ', Xtrain.shape)
    #             print('before transpose: Xtest  shape: ', Xtest.shape)
    #
    #             # 为了适应 pytorch 结构，数据要做 transpose，这里的作用是调整数据的摆放位置
    #             Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    #             Xtrain = Xtrain[:, 0, :, :, :]
    #             Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    #             Xtest = Xtest[:, 0, :, :, :]
    #             print('after transpose: Xtrain shape: ', Xtrain.shape)
    #             print('after transpose: Xtest  shape: ', Xtest.shape)
    #
    #             # 创建 trainloader 和 testloader 将数据放入其中
    #             trainset = TrainDS()
    #             testset = TestDS()
    #
    #             train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True,
    #                                                        num_workers=2)
    #             test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False, num_workers=2)
    #             image_patch = X[i:i + patch_size, j:j + patch_size, :]
    #             image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2])
    #             X_test_image = torch.FloatTensor(image_patch.transpose(0, 3, 1, 2))
    #             X_test_image = X_test_image.to(device)
    #             prediction = net(X_test_image)
    #             prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
    #             outputs[i][j] = prediction + 1
    #
    #             if count == 0:
    #                 All_ytest = [y[i, j]]
    #                 All_y_pred_test = prediction
    #                 count = 1
    #             else:
    #                 All_ytest = np.concatenate((All_ytest, [y[i, j]]))
    #                 All_y_pred_test = np.concatenate((All_y_pred_test, prediction))
    #     if i % 20 == 0:
    #         print('... ... row ', i, ' handling ... ...')
    # np.save('resultNpy/saveoutput1', outputs)
    # cv2.imwrite("figure/finalpre1" + ".jpg", outputs)
    #
    # # 生成分类报告
    # classification = classification_report(All_ytest, All_y_pred_test, digits=4)
    # sklearnSaveReport(All_ytest, All_y_pred_test)

    #
    # predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(5, 5))






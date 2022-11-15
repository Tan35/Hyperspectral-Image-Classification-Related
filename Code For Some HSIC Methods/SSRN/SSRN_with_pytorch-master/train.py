import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import torch
import torch.nn as nn
import  math
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
from torch.autograd import Variable

device =  "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices

def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

class Hsi_Dataset(Dataset):
    def __init__(self,image,label):
        self.image = image
        self.label = label
    def __getitem__(self, index):
        return self.image[index],self.label[index]
    def __len__(self):
        return len(self.image)

class FDSSN(nn.Module):
    def __init__(self,input_dim,data_dim,growth_rate):
        super(FDSSN,self).__init__()
        self.conv3d1 = nn.Conv3d(1,out_channels=24,kernel_size=(7,1,1),stride=(2,1,1))
        self.bn_prelu = nn.Sequential(
            nn.BatchNorm3d(60),
            nn.PReLU())
        self.spectral_conv1 = nn.Sequential(
            nn.BatchNorm3d(24),
            nn.PReLU(),
            nn.Conv3d(in_channels=24,out_channels=growth_rate,kernel_size=(7,1,1),stride=(1,1,1),padding=(3,0,0)))
        self.spectral_conv2 = nn.Sequential(
            nn.BatchNorm3d(36),
            nn.PReLU(),
            nn.Conv3d(in_channels=36, out_channels=growth_rate, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)))
        self.spectral_conv3 = nn.Sequential(
            nn.BatchNorm3d(48),
            nn.PReLU(),
            nn.Conv3d(in_channels=48, out_channels=growth_rate, kernel_size=(7, 1, 1), stride=(1, 1, 1), padding=(3, 0, 0)))

        self.tran1 = nn.Sequential(
            nn.Conv3d(60, data_dim, kernel_size=(97, 1, 1)),
            nn.BatchNorm3d(200),
            nn.PReLU(),
            )

        self.conv3d2 = nn.Conv3d(1, out_channels=24, kernel_size=(200, 3, 3), stride=(1, 1, 1))
        self.spatal_conv1 = nn.Sequential(
            nn.BatchNorm3d(24),
            nn.PReLU(),
            nn.Conv3d(in_channels=24, out_channels=growth_rate, kernel_size=(1, 3, 3), stride=(1, 1, 1),padding=(0,1,1)))
        self.spatal_conv2 = nn.Sequential(
            nn.BatchNorm3d(36),
            nn.PReLU(),
            nn.Conv3d(in_channels=36, out_channels=growth_rate, kernel_size=(1, 3, 3), stride=(1, 1, 1),padding=(0,1,1)))
        self.spatal_conv3 = nn.Sequential(
            nn.BatchNorm3d(48),
            nn.PReLU(),
            nn.Conv3d(in_channels=48, out_channels=growth_rate, kernel_size=(1, 3, 3), stride=(1, 1, 1),padding=(0,1,1)))

        self.aver_pooling = nn.AvgPool3d(kernel_size=(1,7,7),stride=(1,1,1))
        self.dropout1 = nn.Dropout(0.5)
        self.fc = nn.Linear(60,16)

    def forward(self,x):
        x1_0 = self.conv3d1(x)
        x1_1 = self.spectral_conv1(x1_0)
        x1_1_ =torch.cat((x1_0,x1_1),1)
        x1_2 = self.spectral_conv2(x1_1_)
        x1_2_ = torch.cat((x1_0,x1_1,x1_2),1)
        x1_3 = self.spectral_conv3(x1_2_)
        x1 = torch.cat((x1_0,x1_1,x1_2,x1_3),1)
        x1 = self.bn_prelu(x1)

        tran1 = self.tran1(x1)
        tran2 = tran1.transpose(1,2)

        x2_0 = self.conv3d2(tran2)
        x2_1 = self.spatal_conv1(x2_0)
        x2_1_ =torch.cat((x2_0,x2_1),1)
        x2_2 = self.spatal_conv2(x2_1_)
        x2_2_ = torch.cat((x2_0,x2_1,x2_2),1)
        x2_3 = self.spatal_conv3(x2_2_)
        x2 = torch.cat((x2_0,x2_1,x2_2,x2_3),1)
        x2 = self.bn_prelu(x2)

        pool1 = self.aver_pooling(x2)
        faltten1 = pool1.view(pool1.shape[0],-1)
        dropout1 = self.dropout1(faltten1)
        dense = self.fc(dropout1)

        return dense

def main():
    data_IN = sio.loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
    gt_IN = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']
    print(data_IN.shape)
    new_gt_IN = gt_IN

    batch_size = 16
    nb_classes = 16
    nb_epoch = 200  # 400
    img_rows, img_cols = 9, 9
    patience = 200
    INPUT_DIMENSION_CONV = 200
    INPUT_DIMENSION = 200
    # 学习率调到0.0001比较好收敛
    # 训练次数最好稍微增加到70这样子看下识别率
    lr = 0.0001
    TOTAL_SIZE = 10249
    VAL_SIZE = 1025

    TRAIN_SIZE = 2055
    EPOCHES = 50
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    VALIDATION_SPLIT = 0.8  # 20% for trainnig and 80% for validation and testing
    img_channels = 200
    PATCH_LENGTH = 4  # Patch_size (13*2+1)*(13*2+1)

    data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
    gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

    data = preprocessing.scale(data)
    data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
    whole_data = data_
    padded_data = zeroPadding_3D(whole_data, PATCH_LENGTH)

    train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
    test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))

    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    y_train = gt[train_indices] - 1
    # y_train = torch.zeros(len(train_indices),nb_classes).scatter_(1, torch.Tensor(y_train).long().view(-1, 1), 1).to(device)
    y_test = gt[test_indices] - 1
    # y_test = torch.zeros(len(test_indices), nb_classes).scatter_(1, torch.Tensor(y_test).long().view(-1, 1), 1).to(device)

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2],
                                 INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]
    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    train_dataset = Hsi_Dataset(x_train,y_train)
    test_dataset = Hsi_Dataset(x_test,y_test)
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)
    model = FDSSN(1,INPUT_DIMENSION_CONV,12)
    model = model.train(mode=True)
    model = model.apply(weights_init).to(device)

    crossentropy = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),lr=0.0001)
    for epoch in range(EPOCHES):
        running_loss = 0.0
        for i ,data in enumerate(train_dataloader):
            image,label = data
            image = image.transpose(1, 3).unsqueeze(1).type(torch.FloatTensor)
            image,label = Variable(image).to(device),Variable(label.long()).to(device)

            output = model(image).to(device)
            loss = crossentropy(output,label)
            loss.backward()
            optim.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

        with torch.no_grad():
            correct = 0
            total = 0
            best_acc = 0
            for data in test_dataloader:
                image, label = data
                image = image.transpose(1, 3).unsqueeze(1).type(torch.FloatTensor)
                image, label = Variable(image).to(device), Variable(label.long()).to(device)
                outputs = model(image)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum()
            current_acc = (100 * correct / total)
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, current_acc))
        if(current_acc > best_acc):
            best_acc = current_acc
            print('Save the best model.')
            torch.save(model.state_dict(), 'FDSSN.pth')



if __name__ =="__main__":
    main()

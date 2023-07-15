import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from utiles import *
import json
from model import SS_ConvNeXt
import time
from parameters import Parameters
from thop import profile
import pdb
from torchsummary import summary
from thop import clever_format

#-------
# indian pines   patch_size:9  batch size:16
# pavia  patch_size:9  batch size:32
# WHU-HI-HongHu  patch_size:13  batch size:64
# WHU-HI-HanChuan patch_size:13  batch size:64

same_seeds(0)

# 读取配置文件，这里选择使用 json 文件读取(config.json)得到 config 字典变量
with open("config.json", "r") as f:
    config = json.load(f)

# 从 config 字典变量中读取我们的参数
#**********load Hyperparameter*********#
data_name = config["data_name"] #数据集名字
patch_size = config["train"]["patch_size"] # 训练部分的 patch_size
lr = config["train"]["lr"] # 训练部分的学习率
epoch = config["train"]["epoch"] # 训练部分的epoch数
batch_size = config["train"]["batch_size"] # 训练部分的 batch_size
n_classes = config["train"]["num_class"] # 训练部分的数据集的地物种类数（最终要分类的数目）
# **********Load data**********#

print("Load data......")
data_dir = config["data_path"] # 目标数据集路径
target_dir = config["target_path"] # 目标数据集标签路径

# 假设读取的数据集为 Indian Pines
data, target = read_data(data_dir, target_dir) # 通过 read_data 函数得到归一化且将光谱维度调到第一维度的数据和标签（未做任何处理）
# 此时 data.shape = (200, 145, 145) 
input_dim = data.shape[0] # input_dim = 200
input_shape = [batch_size, input_dim, patch_size, patch_size] # input_shape = [batch_size,200,patch_size,patch_size]
# Indian Pines 则 input_shape = [16,200,9,9]

#**********make train val test mask**********#
def mask_choose(target, config):

    mask_type = config["mask_para"]["mask_type_choose"]
    assert isinstance(mask_type, str) and mask_type in ['Proportions', 'Fixed', 'Per_class']

    train_mask = None
    val_mask = None
    test_mask = None

    if mask_type == "Proportions": # 如果是按比例划分
        train_prop = config["mask_para"]["Proportions"]["train_prop"] #0.05
        val_prop = config["mask_para"]["Proportions"]["val_prop"] #0.01
        train_mask, val_mask, test_mask = get_mask(target, train_prop, val_prop) # target 标签，0.05，0.01
        # get_mask 生成用于数据集划分的掩码（mask），跳转看看，还挺有意思

    elif mask_type == "Fixed": # 如果是按固定样本划分
        train_num = config["mask_para"]["Fixed"]["fix_train_num"] # 200
        val_num = config["mask_para"]["Fixed"]["fix_val_num"] # 300
        train_mask, val_mask, test_mask = get_fixed_number_masks(target=target, train_num=train_num, val_num=val_num)

    elif mask_type == "Per_class":
        per_class_num = config["mask_para"]["Per_class"]["class_train_num"]
        val_num = config["mask_para"]["Per_class"]["class_val_num"]
        train_mask, val_mask, test_mask = fixed_class_num_mask(target=target, each_class_num=per_class_num, val_total_num=val_num)
    else:
        print("Please check your mask type choice!")
    sample = np.count_nonzero(train_mask)
    print("The number of training samples have been selected: %d" % np.count_nonzero(train_mask))
    print("The number of validating samples have been selected: %d" % np.count_nonzero(val_mask))
    print("The number of testing samples have been selected: %d" % np.count_nonzero(test_mask))

    return train_mask, val_mask, test_mask, sample

def get_train_val_dataset(data, target, train_mask, val_mask, patch_size):
    train_data, train_target = get_sample(data, target, train_mask, patch_size)
    val_data, val_target = get_sample(data, target, val_mask, patch_size)
    TrainDataset = common_dataset(train_data, train_target, cuda=True)
    ValDataset = common_valdataset(val_data, val_target, cuda=True)

    return TrainDataset, ValDataset

def calculate_param(model, input_shape):
    input = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).float().cuda()
    flops, params = profile(model, inputs=(input,))
    print("FLOPs and Params***************")
    flops, params = clever_format([flops, params], "%.3f")
    print((flops, params))

#*********Get mask data**********#

print("Build train val and test mask......")
train_mask, val_mask, test_mask, sample = mask_choose(target, config)
#**********Get train val dataset

print("Build train and val dataset")
TrainDataset, ValDataset = get_train_val_dataset(data, target, train_mask,
                                                  val_mask, patch_size)
#**********create model*********#
#
# depth = [1,1,3,1]  or  [2,2,6,2]
#
depths = [3, 3, 9, 3]
dims = [64, 128, 256, 512]
print("Creat DS2-cvNet model......")
model = SS_ConvNeXt(input_shape=input_shape, num_classes=n_classes,
                      depths=depths, dims=dims)
model = model.cuda()
calculate_param(model, input_shape)
print("Model paramater.....")

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

if __name__ == '__main__':

    print("Begin training......")
    print("Begin Time:", time.ctime())
    print("Begin training......")
    begin_time = time.time()

    net = train(model, epoch,TrainDataset, ValDataset,
            {'lr': lr, 'optim': optim, 'criterion': criterion}, batch_size)

    end_time = time.time()
    T = end_time-begin_time
    print("Training Time: %.2f" % T)
    print("Predicting......")

    begin_time = time.time()
    data = get_all_patches(data, patch_size=patch_size)
    predict_dataset = PredictionData(data, cuda=True)
    pred = predict(net, predict_dataset, batch_size=batch_size)
    end_time = time.time()
    P = end_time - begin_time
    print("Predicting Time: %.2f" % P)
    pred = pred.reshape(target.shape[0], target.shape[1])

    y_pred_ = pred.copy()
    y_pred_[target == 0] = 0

    pa = Parameters()
    pa.method = 'DS2_cvNet'
    pa.dataset = data_name
    pa.epoch = epoch
    pa.batch_size = batch_size
    pa.depths = depths
    pa.dims = dims
    pa.patch_size = patch_size
    pa.lr = lr
    pa.training_strategy = config["mask_para"]["mask_type_choose"]
    pa.training_num = sample
    pa.training_time = T
    pa.prediction_time = P

    save_dir = './' + data_name + '_result_' + config["mask_para"]["mask_type_choose"] + '/'
    pa.id = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    pa.map_save = os.path.join(save_dir, pa.id) + '/'
    pa.accuracy_save = os.path.join(save_dir, 'accuracy.csv')
    pa.save_params(os.path.join(pa.map_save, 'parameters.json'))
    # sio.savemat([pa.id, pa.map_save, '/prob_map'], pred_map)
    # output map and accuracy curves
    output = ResultOutput(pred, target, y_pred_, train_mask, val_mask, test_mask, pa.get_params())
    output.display_map_and_save()
    output.compute_accuracy_and_save(params_record=['id', 'method', 'dataset', 'training_strategy',
                                                    'training_num', 'batch_size', 'depths',
                                                    'dims', 'patch_size', 'lr', 'epoch',
                                                    'training_time', 'prediction_time'])

    results = metrics(y_pred_, target)
    show_results(results, pa.map_save, agregated=False)
    print("End Time:", time.ctime())

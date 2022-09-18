该项目基于该论文去实现
Hyperspectral Image Classification with Mixed Link Networks
实现日期：2021-06-21
实现作者：li_sample
由于作者在文中没有公布具体的参数，本人根据文章中仅有的部分参数去定义了
可能与原作者存在部分问题
本人主要实现了MLB_A这个block模块，MLB_B这个block模块
MLB_A这个模块没有自动适配数据格式，主要假设数据格式为：
# inputX : [batchSize,x_Inplanes,inputHeigh,inputWidth]
# eg:[128,96,25,25]
MLB_B这个模块可以自动适配数据格式
in the paper ,author set k = 48

主要网络为
一个卷积层得到2k个feature map
3次MLB 的block模块操作，一次增加k个feature map
最后得到5k个feature map
最后的平均池化，
有问题可以互相联系交流，

普通学生实现，可能存在一些问题，勿喷，主要用于学习交流

如果大家引用之后希望在参考文献上附上文章作者的有关内容，
代码参考改网2019MixNet网络，参考Github网址

存在的问题：
关于batchsize过大，但是在单个样本预测数值的时候，由于是一个个数据做预测
导致整个预测结果很差，这里的话可能是因为InputData size 过小导致的
因此，这里决定每个patchsize 设置为25，取代论文中的13*13尺寸
避免过拟合的情况。

但对于在训练和测试同一个batchsize，预测结果不会太差，
如果你知道如何这个问题的产生或者知道如何解决
请留言告诉我，谢谢

模型改进方式：
1.可以在输入图像的时候加入PCA等降维操作，可以提高整个模型的KA
2.一些参数优化大家随自己的喜好去实现，本人因为较忙就不去一一实现了

中文版本翻译：https://www.wzy1999.wang/2021/03/06/109.html

原文提及参数的设置
in the paper ,author set k = 48
我们将批次大小和训练时期都设置为100，然后选择Adam算法[56]来优化所提出的网络。
初始学习率和权重衰减惩罚分别设置为0.001和0.0001。
另外，采用了余弦形状学习速率表，从0.001开始逐渐减小到0。
使用Pytorch框架设计并实现了所提出的网络。

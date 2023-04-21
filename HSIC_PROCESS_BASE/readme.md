## Spectral Signature

+  比较粗糙地实现了几个数据集地物类别的光谱曲线图绘制，在不改动代码的前提下，可实现的数据集有：
  + Indian Pines Dataset (IP)
  + Pavia Centre Dataset (PC)
  + Pavia University Dataset (PU)
  + Salinas Dataset (SA)
  + ~~Kennedy Space Center Dataset (KSC)~~ （数据集比较大，自行下载放到 `./dataset/` 下即可）
+ 当然稍微对代码进行改动也能支持其他数据集，当然前提该数据集是 `.mat` 格式并且你知道该数据集中的地物类别分别是什么
+ 推荐直接运行 `.pynb` 格式文件，此外本代码也提供了是否进行图片保存的选项

---

I have implemented a few spectral curve plots for land cover categories in several datasets. Without changing the code, the following datasets can be handled:

- Indian Pines Dataset (IP)
- Pavia Centre Dataset (PC)
- Pavia University Dataset (PU)
- Salinas Dataset (SA)
- ~~Kennedy Space Center Dataset (KSC)~~ (this dataset are quite large, so please download and put them in the "./dataset/" folder yourself.)

Of course, with slight modifications to the code, other datasets can also be supported, provided that the dataset is in ".mat" format and you know the land cover categories in that dataset.

I recommend running the `.pynb` format file directly. Additionally, the code provides an option to save the image or not.

See u 👾
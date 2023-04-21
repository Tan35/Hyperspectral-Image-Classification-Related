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

![IP_Spectral_Signature_20230421-155407](https://user-images.githubusercontent.com/45460542/233584948-b19fd9a0-ae07-41c8-b090-4c2b6b067f3e.png)
![PC_Spectral_Signature_20230421-155456](https://user-images.githubusercontent.com/45460542/233584991-3bc5fdbc-480f-4783-a9d9-cbff3b462aa3.png)
![PU_Spectral_Signature_20230421-155424](https://user-images.githubusercontent.com/45460542/233585021-1426eed2-e19f-4888-8486-2a32407a24b7.png)
![SA_Spectral_Signature_20230421-155440](https://user-images.githubusercontent.com/45460542/233585047-97500343-e33d-4be0-a219-06cc64bf729d.png)
![KSC_Spectral_Signature_20230421-155508](https://user-images.githubusercontent.com/45460542/233585082-7f167e2a-9fcc-4cd6-87a6-2d658216a805.png)


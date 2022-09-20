# TestDataset_generate
>***用于从照片生成测试的tfrecord文件以及加载模型进行准确度测试***
## 环境依赖
  - 框架：tensorflow2.4
  - CUDA:11.0
  - CUDNN:8.2
  - python:3.7
## 文件介绍
- glass_mask
```bash
里面存放的是生成的tfrecord格式的测试集
```
- TestDataset_generotor.ipynb
```bash
单步调试文件
```
- TestDataset_generotor.py
```bash
用来生成tfrecord测试文件，输入的是对齐后的照片文件夹路径
```
- model_test.py
```bash
加载模型和测试集进行测试
```

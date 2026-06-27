# 基于自主实现 ResNet 的德国交通标志识别

使用 PyTorch 手写 ResNet-34 结构，在 GTSRB 数据集上实现交通标志识别（43 类）。

## 项目结构

```
work/
├── src/
│   ├── model.py          # ResNet-34 手写实现 (BasicBlock+ResNet)
│   └── data_loader.py    # GTSRB 数据集加载与预处理
├── train.py              # 训练脚本
├── test.py               # 全面测试评估脚本（输出各类评价指标与图表）
├── predict.py            # 单张图片预测脚本
├── plot_training_curves.py  # 训练损失/准确率对比曲线生成
├── best_model.pth        # 训练好的模型权重
├── README.md
└── data/
    ├── GTSRB/
    │   ├── Final_Test/       # 测试集图片
    │   └── GT-final_test.csv # 测试集标签
    └── GTSRB_training/
        └── Final_Training/
            └── Images/       # 训练集图片（0~42 编号文件夹）
```

## 代码依赖

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- timm
- matplotlib
- seaborn
- scikit-learn
- pandas
- numpy
- Pillow

安装依赖：

```bash
pip install torch torchvision timm matplotlib seaborn scikit-learn pandas numpy pillow
```

## 数据准备

1. 下载 GTSRB 数据集：
   - 训练集: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
   - 测试集: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
   - 测试集 CSV: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

2. 解压后按以下目录结构放置：

```
data/
├── GTSRB/
│   ├── Final_Test/Images/    # 测试图片 .ppm
│   └── GT-final_test.csv     # 测试标签
└── GTSRB_training/
    └── Final_Training/
        └── Images/           # 00000/ ~ 00042/ 文件夹
```

## 运行步骤

### 1. 训练模型

```bash
python train.py
```

- 默认使用 ResNet-34，num_classes=43，不加载预训练权重
- 如需加载 ImageNet 预训练权重，修改 `train.py` 中 `create_resnet34(pretrained=True)`
- 超参数: batch_size=64, Adam lr=0.001, epochs=10
- 训练完成后自动保存 best_model.pth

### 2. 测试评估

```bash
python test.py
```

输出内容：
- 混淆矩阵（数量图 + 百分比图）
- 分类报告（精确率、召回率、F1）
- ROC 曲线与宏观平均 AUC
- 各类别准确率柱状图
- 置信度分布直方图
- 样本数量与准确率散点图
- 主要混淆类别热力图
- 错误案例分析图（高置信度错误、低置信度正确等）
- 指标汇总表
- classification_report.txt + confusion_matrix.csv

所有图片尺寸为 5.04 英寸宽（A4 可打印宽度的 80%），300 DPI，六号宋体。

### 3. 单张图片预测

```bash
python predict.py <图片路径> [选项]
```

选项：
- `--topk K`      显示前K个预测结果（默认5）
- `--model path`  指定模型权重路径（默认best_model.pth）
- `--save`        保存可视化结果图
- `--save_path`   结果图保存路径

示例：

```bash
python predict.py test.ppm
python predict.py data/GTSRB/Final_Test/Images/00000.ppm --topk 10 --save
```

### 4. 训练曲线对比

```bash
python plot_training_curves.py
```

输出方案A（加载预训练）与方案B（不加载预训练）的训练损失和准确率对比曲线。

## 模型结构

手写 ResNet-34 配置：

| 模块 | 结构 | 输出尺寸 |
|------|------|---------|
| conv1 | 7x7, 64, stride 2 | 112x112 |
| maxpool | 3x3, stride 2 | 56x56 |
| layer1 | BasicBlock x 3, 64 | 56x56 |
| layer2 | BasicBlock x 4, 128 | 28x28 |
| layer3 | BasicBlock x 6, 256 | 14x14 |
| layer4 | BasicBlock x 3, 512 | 7x7 |
| avgpool | adaptive 1x1 | 1x1 |
| fc | 512 -> 43 | 43 |

BasicBlock: 3x3 conv -> BN -> ReLU -> 3x3 conv -> BN -> 残差连接 -> ReLU

## 实验结果

| 指标 | 方案A| 方案B |
|------|-------------------|---------------|
| 测试准确率 | 98.99% | 97.09% |
| Top-5 准确率 | 99.71% | 99.26% |
| 宏观平均 F1 | 0.9853 | 0.9449 |
| 加权平均 F1 | 0.9897 | 0.9704 |
| 收敛至 98% val_acc | 1 epoch | 4 epochs |
| 高置信度错误占比 | 34.4% | 17.2% |

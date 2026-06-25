import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.data_loader import create_test_loader, TEST_IMG_DIR, TEST_CSV_PATH, BASE_DIR
from src.model import create_resnet34

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("正在加载测试集...")
test_loader = create_test_loader(
    img_dir=str(TEST_IMG_DIR),
    csv_path=str(TEST_CSV_PATH),
    batch_size=64
)

model = create_resnet34(num_classes=43, pretrained=False)
model_path = BASE_DIR / 'best_model.pth'
if not model_path.exists():
    raise FileNotFoundError(f"未找到模型文件: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []
print("开始测试集推理...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"测试集最终准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

cm = confusion_matrix(all_labels, all_preds)

cm_percentage = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

plt.figure(figsize=(14, 12))
sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Reds',
            cbar=True, cbar_kws={'label': 'Percentage (%)'},
            xticklabels=range(43), yticklabels=range(43))
plt.title(f'Confusion Matrix (Row Normalized %) - Test Acc: {test_acc:.2%}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix_percentage.png', dpi=300)
print("百分比混淆矩阵已保存为 confusion_matrix_percentage.png")

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True,
            xticklabels=range(43), yticklabels=range(43))
plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2%}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("混淆矩阵图片已保存: confusion_matrix.png")

np.savetxt('confusion_matrix.csv', cm, delimiter=',', fmt='%d',
           header='True\\Pred,' + ','.join([str(i) for i in range(43)]), comments='')
print("混淆矩阵数据表已保存: confusion_matrix.csv")

report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
macro_f1 = report['macro avg']['f1-score']
weighted_f1 = report['weighted avg']['f1-score']

print(f"宏观平均 F1-Score: {macro_f1:.4f}")
print(f"加权平均 F1-Score: {weighted_f1:.4f}")

with open('classification_report.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, zero_division=0))
print("分类报告已保存: classification_report.txt")

np.fill_diagonal(cm, 0)
err_pairs = []
for i in range(43):
    for j in range(43):
        if cm[i, j] > 0:
            err_pairs.append((i, j, cm[i, j]))
err_pairs.sort(key=lambda x: -x[2])

print("最易混淆的 5 对类别 (真实类别 -> 预测类别):")
for k, (true, pred, count) in enumerate(err_pairs[:5]):
    print(f"  {k+1}. 类别 {true} -> 被误判为 类别 {pred}, 共 {count} 次")

print("1. 准确率: {:.2f}%".format(test_acc*100))
print("2. 宏观 F1: {:.4f}".format(macro_f1))
print("3. 混淆矩阵图: confusion_matrix.png")
print("4. 易混淆类别对: 见上方打印")

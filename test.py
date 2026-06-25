import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from src.data_loader import create_test_loader, TEST_IMG_DIR, TEST_CSV_PATH, BASE_DIR
from src.model import create_resnet34
import time

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
all_probs = []
print("开始测试集推理...")
inference_start = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
inference_time = time.time() - inference_start

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

test_acc = np.mean(all_preds == all_labels)
print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"推理耗时: {inference_time:.2f}s, "
      f"平均 {inference_time / len(all_labels) * 1000:.2f}ms/张")

top3_correct = 0
top5_correct = 0
for i in range(len(all_labels)):
    top3 = np.argsort(all_probs[i])[-3:][::-1]
    top5 = np.argsort(all_probs[i])[-5:][::-1]
    if all_labels[i] in top3:
        top3_correct += 1
    if all_labels[i] in top5:
        top5_correct += 1

top3_acc = top3_correct / len(all_labels)
top5_acc = top5_correct / len(all_labels)
print(f"Top-3 准确率: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
print(f"Top-5 准确率: {top5_acc:.4f} ({top5_acc*100:.2f}%)")

cm = confusion_matrix(all_labels, all_preds)
cm_percentage = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

plt.figure(figsize=(14, 12))
sns.heatmap(cm_percentage, annot=False, fmt='.1f', cmap='Reds',
            cbar=True, cbar_kws={'label': '百分比 %'},
            xticklabels=range(43), yticklabels=range(43))
plt.title(f'混淆矩阵 - 测试准确率: {test_acc:.2%}')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.savefig('confusion_matrix_percentage.png', dpi=300)
print("百分比混淆矩阵已保存: confusion_matrix_percentage.png")

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True,
            cbar_kws={'label': '样本数'},
            xticklabels=range(43), yticklabels=range(43))
plt.title(f'混淆矩阵 - 测试准确率: {test_acc:.2%}')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
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

class_correct = np.zeros(43)
class_total = np.zeros(43)
for i in range(len(all_labels)):
    label = all_labels[i]
    class_total[label] += 1
    if all_preds[i] == label:
        class_correct[label] += 1

class_acc = np.divide(class_correct, class_total,
                      out=np.zeros_like(class_correct, dtype=float), where=class_total>0)

plt.figure(figsize=(14, 6))
colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.5 else 'red' for acc in class_acc]
plt.bar(range(43), class_acc * 100, color=colors)
plt.axhline(y=test_acc*100, color='blue', linestyle='--', label=f'总体: {test_acc*100:.1f}%')
plt.xlabel('类别编号')
plt.ylabel('准确率 %')
plt.title('各类别准确率')
plt.xticks(range(0, 43, 5))
plt.legend()
plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=300)
print("每类准确率图已保存: per_class_accuracy.png")

max_probs = np.max(all_probs, axis=1)
conf_correct = max_probs[all_preds == all_labels]
conf_wrong = max_probs[all_preds != all_labels]

plt.figure(figsize=(10, 6))
plt.hist(conf_correct, bins=50, alpha=0.7, label='正确', color='green')
plt.hist(conf_wrong, bins=50, alpha=0.7, label='错误', color='red')
plt.xlabel('置信度')
plt.ylabel('数量')
plt.title('预测置信度分布')
plt.legend()
plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=300)
avg_conf_correct = np.mean(conf_correct) if len(conf_correct) > 0 else 0
avg_conf_wrong = np.mean(conf_wrong) if len(conf_wrong) > 0 else 0
print(f"正确样本平均置信度: {avg_conf_correct:.4f}")
print(f"错误样本平均置信度: {avg_conf_wrong:.4f}")
con_wrong_high = np.mean(conf_wrong > 0.9) if len(conf_wrong) > 0 else 0
print(f"错误样本中高置信度(>0.9)占比: {con_wrong_high*100:.1f}%")
print("置信度分布图已保存: confidence_distribution.png")

plt.figure(figsize=(10, 6))
sc = plt.scatter(class_total, class_acc * 100, alpha=0.6, c=range(43), cmap='tab20', s=80)
for i in range(43):
    plt.annotate(str(i), (class_total[i], class_acc[i] * 100), fontsize=8, alpha=0.8)
plt.xlabel('测试样本数')
plt.ylabel('准确率 %')
plt.title('样本数量与准确率关系')
plt.tight_layout()
plt.savefig('support_vs_accuracy.png', dpi=300)
print("支持数与准确率散点图已保存: support_vs_accuracy.png")

np.fill_diagonal(cm, 0)
err_pairs = []
for i in range(43):
    for j in range(43):
        if cm[i, j] > 0:
            err_pairs.append((i, j, cm[i, j]))
err_pairs.sort(key=lambda x: -x[2])

top_confused = set()
for true_label, pred_label, _ in err_pairs[:15]:
    top_confused.add(true_label)
    top_confused.add(pred_label)
top_confused = sorted(top_confused)

cm_focused = confusion_matrix(all_labels, all_preds, labels=top_confused)
cm_focused_pct = cm_focused.astype('float') / (cm_focused.sum(axis=1, keepdims=True) + 1e-9) * 100

n_focused = len(top_confused)
figsize_focused = max(6, n_focused * 0.6)
plt.figure(figsize=(figsize_focused, figsize_focused * 0.9))
sns.heatmap(cm_focused_pct, annot=True, fmt='.1f', cmap='Reds',
            cbar=True, cbar_kws={'label': '百分比 %'},
            xticklabels=top_confused, yticklabels=top_confused,
            annot_kws={'size': 8})
plt.title('主要混淆类别分布')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.savefig('confusion_top_confusions.png', dpi=300)
print("主要混淆类别热力图已保存: confusion_top_confusions.png")

print("最易混淆的 5 对类别 (真实 -> 预测):")
for k, (true, pred, count) in enumerate(err_pairs[:5]):
    print(f"  {k+1}. 类别 {true} -> 类别 {pred}, 共 {count} 次")

worst_idx = np.argsort(class_acc)[:5]
print("准确率最低的 5 个类别:")
for i in worst_idx:
    print(f"  类别 {i}: {class_acc[i]*100:.1f}% ({int(class_correct[i])}/{int(class_total[i])})")

print(f"\n准确率: {test_acc*100:.2f}%")
print(f"Top-3: {top3_acc*100:.2f}% | Top-5: {top5_acc*100:.2f}%")
print(f"宏观 F1: {macro_f1:.4f} | 加权 F1: {weighted_f1:.4f}")
print(f"推理: {inference_time:.2f}s 总 | {inference_time/len(all_labels)*1000:.2f}ms/张")
print(f"结果文件: confusion_matrix.png, per_class_accuracy.png, "
      f"confidence_distribution.png, support_vs_accuracy.png, "
      f"confusion_top_confusions.png")

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from src.data_loader import create_test_loader, TEST_IMG_DIR, TEST_CSV_PATH, BASE_DIR
from src.model import create_resnet34
import time

plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 7.5

FIG_WIDTH = 5.04

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
sampled_images = {}
MAX_ERROR_SAMPLES = 80
MAX_CORRECT_SAMPLES = 30
error_count = 0
correct_count = 0

print("开始测试集推理...")
inference_start = time.time()
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

        batch_size = labels.size(0)
        for i in range(batch_size):
            global_idx = batch_idx * batch_size + i
            is_correct = (predicted[i] == labels[i]).item()
            conf = probs[i, predicted[i]].item()
            if not is_correct and error_count < MAX_ERROR_SAMPLES:
                sampled_images[global_idx] = (
                    images[i].cpu(), labels[i].item(), predicted[i].item(), conf
                )
                error_count += 1
            elif is_correct and correct_count < MAX_CORRECT_SAMPLES:
                sampled_images[global_idx] = (
                    images[i].cpu(), labels[i].item(), predicted[i].item(), conf
                )
                correct_count += 1

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

# 混淆矩阵 - 百分比
plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.86))
sns.heatmap(cm_percentage, annot=False, fmt='.1f', cmap='Reds',
            cbar=True, cbar_kws={'label': '百分比 %'},
            xticklabels=range(43), yticklabels=range(43))
plt.title(f'混淆矩阵 - 测试准确率: {test_acc:.2%}')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.savefig('confusion_matrix_percentage.png', dpi=300)
print("百分比混淆矩阵已保存: confusion_matrix_percentage.png")

# 混淆矩阵 - 数量
plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.86))
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

# 分类报告
report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
macro_f1 = report['macro avg']['f1-score']
weighted_f1 = report['weighted avg']['f1-score']

print(f"宏观平均 F1-Score: {macro_f1:.4f}")
print(f"加权平均 F1-Score: {weighted_f1:.4f}")

with open('classification_report.txt', 'w') as f:
    f.write(classification_report(all_labels, all_preds, zero_division=0))
print("分类报告已保存: classification_report.txt")

# 各类别详细性能
print("\n各类别详细性能:")
print(f"{'类别':>5} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'样本数':>6}")
print("-" * 45)
class_correct = np.zeros(43)
class_total = np.zeros(43)
for i in range(len(all_labels)):
    label = all_labels[i]
    class_total[label] += 1
    if all_preds[i] == label:
        class_correct[label] += 1
class_acc = np.divide(class_correct, class_total,
                      out=np.zeros_like(class_correct, dtype=float), where=class_total>0)
for i in range(43):
    cls = report.get(str(i), {})
    prec = cls.get('precision', 0)
    rec = cls.get('recall', 0)
    f1 = cls.get('f1-score', 0)
    support = cls.get('support', 0)
    print(f"{i:>5} {class_acc[i]*100:>7.2f}% {prec*100:>7.2f}% {rec*100:>7.2f}% {f1:>8.4f} {int(support):>6}")

# 敏感性与特异性
print("\n敏感性 Sensitivity 与 特异性 Specificity:")
print(f"{'类别':>5} {'敏感性':>10} {'特异性':>10}")
print("-" * 27)
total_samples = len(all_labels)
sensitivities = []
specificities = []
for i in range(43):
    tp = cm[i, i]
    fn = class_total[i] - tp
    fp = cm[:, i].sum() - tp
    tn = total_samples - tp - fn - fp
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    print(f"{i:>5} {sensitivity*100:>9.2f}% {specificity*100:>9.2f}%")

macro_sensitivity = np.mean(sensitivities)
macro_specificity = np.mean(specificities)
print(f"\n宏观平均敏感性: {macro_sensitivity*100:.2f}%")
print(f"宏观平均特异性: {macro_specificity*100:.2f}%")

# ROC 曲线与 AUC
print("\n正在计算 ROC 曲线与 AUC 值...")
n_classes = 43
y_true_bin = label_binarize(all_labels, classes=range(n_classes))

sorted_idx = np.argsort(class_acc)
worst_idx = sorted_idx[:3]
best_idx = sorted_idx[-3:]
plot_classes = list(worst_idx) + list(best_idx)

plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.70))
fpr_macro_list = []
tpr_macro_list = []
auc_scores = []

for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)
    fpr_grid = np.linspace(0, 1, 1000)
    tpr_interp = np.interp(fpr_grid, fpr[::-1], tpr[::-1])
    fpr_macro_list.append(fpr_grid)
    tpr_macro_list.append(tpr_interp)
    if i in plot_classes:
        label = f"类别 {i} AUC={roc_auc:.3f}"
        lw = 2 if i in worst_idx else 1.5
        linestyle = '--' if i in worst_idx else '-'
        plt.plot(fpr, tpr, lw=lw, linestyle=linestyle, label=label)

mean_tpr = np.mean(tpr_macro_list, axis=0)
macro_auc = auc(fpr_grid, mean_tpr)
plt.plot(fpr_grid, mean_tpr, 'k-', lw=2.5,
         label=f'宏观平均 AUC={macro_auc:.3f}')

plt.plot([0, 1], [0, 1], 'gray', lw=1, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 False Positive Rate')
plt.ylabel('真阳性率 True Positive Rate')
plt.title('ROC 曲线')
plt.legend(loc="lower right", fontsize=7.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
print("ROC 曲线已保存: roc_curve.png")
print(f"宏观平均 AUC: {macro_auc:.4f}")

print(f"\n各类别 AUC 值:")
auc_by_class = sorted([(i, auc_scores[i]) for i in range(n_classes)], key=lambda x: -x[1])
print(f"{'类别':>5} {'AUC':>8}")
print("-" * 15)
for i, a in auc_by_class:
    print(f"{i:>5} {a:>8.4f}")
print(f"宏观平均 AUC: {np.mean(auc_scores):.4f}")

# 各类别准确率柱状图
plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.43))
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

# 置信度分布
max_probs = np.max(all_probs, axis=1)
conf_correct = max_probs[all_preds == all_labels]
conf_wrong = max_probs[all_preds != all_labels]

plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.60))
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

# 样本数量与准确率关系
plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.60))
scatter = plt.scatter(class_total, class_acc * 100, alpha=0.6, c=range(43), cmap='tab20', s=80)
for i in range(43):
    plt.annotate(str(i), (class_total[i], class_acc[i] * 100), fontsize=7.5, alpha=0.8)
plt.xlabel('测试样本数')
plt.ylabel('准确率 %')
plt.title('样本数量与准确率关系')
plt.tight_layout()
plt.savefig('support_vs_accuracy.png', dpi=300)
print("支持数与准确率散点图已保存: support_vs_accuracy.png")

# 最易混淆类别对
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
fig_w = min(FIG_WIDTH, n_focused * 0.50)
plt.figure(figsize=(fig_w, fig_w * 0.90))
sns.heatmap(cm_focused_pct, annot=True, fmt='.1f', cmap='Reds',
            cbar=True, cbar_kws={'label': '百分比 %'},
            xticklabels=top_confused, yticklabels=top_confused,
            annot_kws={'size': 7.5})
plt.title('主要混淆类别分布')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.tight_layout()
plt.savefig('confusion_top_confusions.png', dpi=300)
print("主要混淆类别热力图已保存: confusion_top_confusions.png")

# 错误案例分析
print("\n正在生成错误案例可视化分析...")

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

class_names = [
    "限速20", "限速30", "限速50", "限速60", "限速70", "限速80",
    "解除80", "限速100", "限速120", "禁止超车", "禁止货车超车",
    "交叉口优先", "优先道路", "让行", "停车让行", "禁止通行",
    "禁止货车", "禁止驶入", "注意危险", "左弯危险", "右弯危险",
    "连续弯路", "路面不平", "路面湿滑", "右侧变窄", "道路施工",
    "信号灯", "注意行人", "注意儿童", "注意自行车", "注意冰雪",
    "野生动物", "解除限禁", "前方右转", "前方左转", "直行",
    "直行或右转", "直行或左转", "靠右行驶", "靠左行驶", "环岛",
    "解除禁超", "解除禁货车超"
]

SAMPLES_PER_GRID = 6

def save_case_grid(indices, title, filename, max_samples=SAMPLES_PER_GRID):
    available = [i for i in indices if i in sampled_images]
    n = min(len(available), max_samples)
    if n == 0:
        print(f"  跳过 {filename}: 无可用图片")
        return
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(FIG_WIDTH, FIG_WIDTH * 0.60))
    axes = axes.flatten()

    for plot_idx in range(n):
        ax = axes[plot_idx]
        global_idx = available[plot_idx]
        img_tensor, true_label, pred_label, conf = sampled_images[global_idx]
        img = denormalize(img_tensor)
        img_np = img.permute(1, 2, 0).numpy()
        ax.imshow(img_np)

        is_correct = true_label == pred_label
        if is_correct:
            title_text = f"正确: 类别{true_label}\n{class_names[true_label]}\n置信度 {conf:.2%}"
            border_color = 'green'
        else:
            title_text = (f"真实{true_label}->预测{pred_label}\n"
                          f"{class_names[true_label]}->{class_names[pred_label]}\n"
                          f"置信度 {conf:.2%}")
            border_color = 'red'
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2)
        ax.set_title(title_text, fontsize=7.5, color=border_color)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')
    plt.suptitle(title, fontsize=7.5, y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {filename}")

if len(sampled_images) >= 6:
    all_stored_indices = list(sampled_images.keys())

    stored_errors = [idx for idx in all_stored_indices
                     if sampled_images[idx][1] != sampled_images[idx][2]]
    stored_correct = [idx for idx in all_stored_indices
                      if sampled_images[idx][1] == sampled_images[idx][2]]

    stored_errors.sort(key=lambda x: -sampled_images[x][3])
    stored_errors_low = list(reversed(stored_errors))
    stored_correct.sort(key=lambda x: -sampled_images[x][3])
    stored_correct_low = list(reversed(stored_correct))

    if len(stored_errors) > 0:
        save_case_grid(stored_errors, "高置信度错误案例 模型很自信但预测错误",
                       "error_high_confidence.png")
    if len(stored_correct_low) > 0:
        save_case_grid(stored_correct_low, "低置信度正确案例 模型不太确定但实际正确",
                       "correct_low_confidence.png")
    if len(stored_correct) > 0:
        save_case_grid(stored_correct, "高置信度正确案例",
                       "correct_high_confidence.png")

    top_err_types = err_pairs[:3]
    typical_indices = []
    for true_cls, pred_cls, _ in top_err_types:
        count = 0
        for idx in all_stored_indices:
            img_t, true_l, pred_l, _ = sampled_images[idx]
            if true_l == true_cls and pred_l == pred_cls:
                typical_indices.append(idx)
                count += 1
                if count >= 2:
                    break
    if typical_indices:
        save_case_grid(typical_indices,
                       "典型错误案例 最常见混淆类型",
                       "error_typical_cases.png",
                       max_samples=len(typical_indices))
else:
    print("采样的图片数量不足，跳过可视化分析")

# 指标汇总表格
print("\n生成指标汇总表...")
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_WIDTH * 0.60))
ax.axis('tight')
ax.axis('off')

table_data = [
    ["测试集样本数", str(len(all_labels))],
    ["准确率", f"{test_acc*100:.2f}%"],
    ["Top-3 准确率", f"{top3_acc*100:.2f}%"],
    ["Top-5 准确率", f"{top5_acc*100:.2f}%"],
    ["宏观平均精确率", f"{report['macro avg']['precision']*100:.2f}%"],
    ["宏观平均召回率", f"{report['macro avg']['recall']*100:.2f}%"],
    ["宏观平均 F1", f"{report['macro avg']['f1-score']:.4f}"],
    ["加权平均 F1", f"{report['weighted avg']['f1-score']:.4f}"],
    ["宏观平均敏感性", f"{macro_sensitivity*100:.2f}%"],
    ["宏观平均特异性", f"{macro_specificity*100:.2f}%"],
    ["宏观平均 AUC", f"{macro_auc:.4f}"],
    ["推理总耗时", f"{inference_time:.2f}s"],
    ["平均推理速度", f"{inference_time/len(all_labels)*1000:.2f}ms/张"],
    ["正确样本平均置信度", f"{avg_conf_correct:.4f}"],
    ["错误样本平均置信度", f"{avg_conf_wrong:.4f}"],
    ["高置信度错误占比", f"{con_wrong_high*100:.1f}%"],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                 colWidths=[0.35, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1, 1.6)
for (row, col), cell in table.get_celld().items():
    if col == 0:
        cell.set_facecolor('#f0f0f0')
        cell.set_text_props(fontweight='normal')
ax.set_title('模型性能指标汇总', fontsize=7.5, pad=20)
fig.tight_layout()
fig.savefig('metrics_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("指标汇总表已保存: metrics_summary.png")

# 输出最易混淆的 5 对
print("\n最易混淆的 5 对类别 (真实 -> 预测):")
for k, (true, pred, count) in enumerate(err_pairs[:5]):
    print(f"  {k+1}. 类别 {true} -> 类别 {pred}, 共 {count} 次")

print("准确率最低的 5 个类别:")
for i in sorted_idx[:5]:
    print(f"  类别 {i}: {class_acc[i]*100:.1f}% ({int(class_correct[i])}/{int(class_total[i])})")

print(f"\n准确率: {test_acc*100:.2f}%")
print(f"Top-3: {top3_acc*100:.2f}% | Top-5: {top5_acc*100:.2f}%")
print(f"宏观 F1: {macro_f1:.4f} | 加权 F1: {weighted_f1:.4f}")
print(f"推理: {inference_time:.2f}s 总 | {inference_time/len(all_labels)*1000:.2f}ms/张")

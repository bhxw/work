"""
训练曲线对比图生成脚本
方案0（加载预训练权重）与方案1（不加载预训练权重）对比
输出两种图片：
  1. training_loss_comparison.png   — 训练损失对比曲线
  2. training_accuracy_comparison.png — 训练/验证准确率对比曲线
"""

import matplotlib.pyplot as plt
import numpy as np

# 尝试设置中文字体 SimSun，备选 SimHei / Microsoft YaHei / DejaVu Sans
for font_name in ['SimSun', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']:
    try:
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 7.5
        plt.rcParams['font.weight'] = 'normal'
        break
    except:
        continue

# A4 可打印宽度: 21cm - 2.5cm*2 边距 = 16cm = 6.3 英寸
# 要求占宽约 50%，取 3.15 英寸
FIG_WIDTH = 3.15
LOSS_HEIGHT = 2.4   # 损失图高度，保持比例
ACC_HEIGHT = 2.7    # 准确率图高度，保持比例
DPI = 300

# ---- 数据 ----------------------------------------------------------------
epochs = np.arange(1, 11)

# 方案0（加载预训练权重）
loss_0      = [0.4092, 0.0256, 0.0165, 0.0104, 0.0109, 0.0155, 0.0114, 0.0125, 0.0022, 0.0054]
train_acc_0 = [0.8952, 0.9935, 0.9952, 0.9973, 0.9975, 0.9963, 0.9974, 0.9972, 0.9996, 0.9989]
val_acc_0   = [0.9927, 0.9952, 0.9879, 0.9948, 0.9992, 0.9939, 0.9953, 0.9991, 0.9996, 0.9986]

# 方案1（不加载预训练权重）
loss_1      = [2.3105, 0.5051, 0.0890, 0.0449, 0.0439, 0.0315, 0.0403, 0.0214, 0.0268, 0.0198]
train_acc_1 = [0.3037, 0.8392, 0.9744, 0.9868, 0.9870, 0.9907, 0.9881, 0.9936, 0.9918, 0.9938]
val_acc_1   = [0.5020, 0.9169, 0.9549, 0.9825, 0.9853, 0.9925, 0.9901, 0.9903, 0.9955, 0.9957]

# 颜色方案
C0 = '#E74C3C'   # 方案0 主色（红）
C1 = '#3498DB'   # 方案1 主色（蓝）

# =========================================================================
# 图1：训练损失对比
# =========================================================================
fig1, ax1 = plt.subplots(figsize=(FIG_WIDTH, LOSS_HEIGHT))

ax1.plot(epochs, loss_0, 'o-', color=C0, linewidth=1.0, markersize=3.5,
         markerfacecolor=C0, markeredgewidth=0.5, markeredgecolor='white',
         label='方案A')
ax1.plot(epochs, loss_1, 's-', color=C1, linewidth=1.0, markersize=3.5,
         markerfacecolor=C1, markeredgewidth=0.5, markeredgecolor='white',
         label='方案B')

ax1.set_xlabel('训练轮次', fontsize=7.5, fontweight='normal')
ax1.set_ylabel('损失值', fontsize=7.5, fontweight='normal')
ax1.set_title('训练损失对比', fontsize=7.5, fontweight='normal')
ax1.legend(fontsize=7.5, frameon=True, fancybox=False, edgecolor='gray')
ax1.set_xticks(epochs)
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.tick_params(labelsize=7.5)

plt.tight_layout(pad=0.5)
plt.savefig('training_loss_comparison.png', dpi=DPI, bbox_inches='tight')
plt.savefig('training_loss_comparison.eps', format='eps', bbox_inches='tight')
plt.close()
print('已保存: training_loss_comparison.png / .eps')

# =========================================================================
# 图2：训练/验证准确率对比
# =========================================================================
fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, ACC_HEIGHT))

ax2.plot(epochs, train_acc_0, 'o-',  color=C0, linewidth=1.0, markersize=3.5,
         markerfacecolor=C0, markeredgewidth=0.5, markeredgecolor='white',
         label='方案A 训练准确率')
ax2.plot(epochs, val_acc_0,   'o--', color=C0, linewidth=1.0, markersize=3.5,
         markerfacecolor='white', markeredgewidth=0.5, markeredgecolor=C0,
         alpha=0.7, label='方案A 验证准确率')
ax2.plot(epochs, train_acc_1, 's-',  color=C1, linewidth=1.0, markersize=3.5,
         markerfacecolor=C1, markeredgewidth=0.5, markeredgecolor='white',
         label='方案B 训练准确率')
ax2.plot(epochs, val_acc_1,   's--', color=C1, linewidth=1.0, markersize=3.5,
         markerfacecolor='white', markeredgewidth=0.5, markeredgecolor=C1,
         alpha=0.7, label='方案B 验证准确率')

ax2.set_xlabel('训练轮次', fontsize=7.5, fontweight='normal')
ax2.set_ylabel('准确率', fontsize=7.5, fontweight='normal')
ax2.set_title('训练与验证准确率对比', fontsize=7.5, fontweight='normal')
ax2.legend(fontsize=6.5, frameon=True, fancybox=False, edgecolor='gray',
           ncol=2, loc='lower right')
ax2.set_xticks(epochs)
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.25, linestyle='--')
ax2.tick_params(labelsize=7.5)

plt.tight_layout(pad=0.5)
plt.savefig('training_accuracy_comparison.png', dpi=DPI, bbox_inches='tight')
plt.savefig('training_accuracy_comparison.eps', format='eps', bbox_inches='tight')
plt.close()
print('已保存: training_accuracy_comparison.png / .eps')

print('\n两张对比图生成完毕，可直接以 100% 比例插入 A4 Word 文档。')

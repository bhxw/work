"""
单张图片预测脚本
用法: python predict.py <图片路径>
支持格式: 任意 PIL 可读格式 ppm jpg png bmp jpeg tiff 等

示例:
  python predict.py test.png
  python predict.py data/GTSRB/Final_Test/Images/00000.ppm
  python predict.py my_photo.jpg --save
  python predict.py my_photo.jpg --topk 10
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from src.model import create_resnet34

# GTSRB 43 类交通标志名称映射
GTSRB_CLASSES = [
    "限速 20km/h",
    "限速 30km/h",
    "限速 50km/h",
    "限速 60km/h",
    "限速 70km/h",
    "限速 80km/h",
    "解除限速 80km/h",
    "限速 100km/h",
    "限速 120km/h",
    "禁止超车",
    "禁止 3.5吨以上车辆超车",
    "前方交叉口优先通行",
    "优先道路",
    "让行",
    "停车让行",
    "禁止所有车辆通行",
    "禁止 3.5吨以上车辆通行",
    "禁止驶入",
    "注意危险",
    "左转弯危险",
    "右转弯危险",
    "连续弯路",
    "路面不平",
    "路面湿滑",
    "右侧变窄",
    "道路施工",
    "交通信号灯",
    "注意行人",
    "注意儿童",
    "注意自行车",
    "注意冰雪",
    "注意野生动物",
    "解除所有限速与禁止超车",
    "前方右转",
    "前方左转",
    "直行",
    "直行或右转",
    "直行或左转",
    "靠右行驶",
    "靠左行驶",
    "环岛行驶",
    "解除禁止超车",
    "解除禁止 3.5吨以上车辆超车",
]


def preprocess_image(img_path):
    """
    加载任意格式图片，转为 RGB，Resize 到 224x224，归一化
    返回 input_tensor 和原始 PIL 图片
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"图片文件不存在: {img_path}")

    pil_img = Image.open(img_path).convert("RGB")
    original = pil_img.copy()

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_img).unsqueeze(0)
    return input_tensor, original


def predict_image(model, img_path, device, topk=5):
    """对单张图片推理，返回预测类别、置信度、Top-K 列表和原始图片"""
    input_tensor, original = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        topk_probs, topk_indices = torch.topk(probs, topk, dim=1)

    pred_class_id = topk_indices[0, 0].item()
    confidence = topk_probs[0, 0].item()
    topk_ids = topk_indices[0].cpu().tolist()
    topk_confs = topk_probs[0].cpu().tolist()

    return pred_class_id, confidence, topk_ids, topk_confs, original


def display_results(pred_id, confidence, topk_ids, topk_confs, original_img, img_path, topk=5):
    """打印预测结果"""
    img_name = os.path.basename(img_path)
    original_size = original_img.size

    print(f"\n图片: {img_name}  尺寸: {original_size[0]}x{original_size[1]}")
    print(f"预测类别: {pred_id} — {GTSRB_CLASSES[pred_id]}")
    print(f"置信度: {confidence:.4f}  {confidence*100:.2f}%")
    print()

    print(f"Top-{topk} 预测:")
    for i, (cid, cf) in enumerate(zip(topk_ids, topk_confs)):
        print(f"  #{i+1}  类别 {cid:2d}  {cf*100:6.2f}%   {GTSRB_CLASSES[cid]}")
        if i == 0:
            print(f"      最终预测")
        if cf < 0.01:
            print(f"      余下类别置信度过低，不再显示")
            break

    max_conf = max(topk_confs)
    second_conf = topk_confs[1] if len(topk_confs) > 1 else 0
    margin = max_conf - second_conf
    print(f"\n置信度分析:")
    print(f"  最高置信度: {max_conf:.4f}  {max_conf*100:.2f}%")
    print(f"  次高置信度: {second_conf:.4f}  {second_conf*100:.2f}%")
    print(f"  置信度差距: {margin:.4f}  {margin*100:.2f}%")
    if confidence > 0.9:
        print(f"  判断: 模型对该结果高度自信")
    elif confidence > 0.7:
        print(f"  判断: 模型对该结果较为自信")
    else:
        print(f"  判断: 模型对该结果不太确定")

    return {
        "image": img_name,
        "image_size": original_size,
        "predicted_class_id": pred_id,
        "predicted_class_name": GTSRB_CLASSES[pred_id],
        "confidence": confidence,
        "topk": [
            {"class_id": cid, "class_name": GTSRB_CLASSES[cid], "confidence": cf}
            for cid, cf in zip(topk_ids, topk_confs)
        ],
        "confidence_margin": margin,
    }


def save_visualization(result, original_img, save_path="prediction_result.png"):
    """保存可视化结果图"""
    try:
        import matplotlib.pyplot as plt

        for font_name in ['SimSun', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']:
            try:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                plt.rcParams['font.size'] = 7.5
                break
            except:
                continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.imshow(original_img)
        ax1.set_title(f"预测: {result['predicted_class_name']}\n"
                      f"类别 {result['predicted_class_id']} | "
                      f"置信度 {result['confidence']*100:.1f}%",
                      fontsize=7.5)
        ax1.axis("off")

        topk = result["topk"][:10]
        names = [f"{t['class_id']}\n{t['class_name'][:6]}" for t in topk]
        confs = [t["confidence"] * 100 for t in topk]
        colors = ["#2196F3" if i > 0 else "#FF5722" for i in range(len(topk))]
        bars = ax2.bar(range(len(topk)), confs, color=colors, alpha=0.8)
        ax2.set_xticks(range(len(topk)))
        ax2.set_xticklabels(names, fontsize=7.5)
        ax2.set_ylabel("置信度 %")
        ax2.set_title("Top-K 预测置信度", fontsize=7.5)
        ax2.set_ylim(0, 105)
        for bar, conf in zip(bars, confs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{conf:.1f}%", ha="center", va="bottom", fontsize=7.5)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"结果图已保存: {save_path}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="GTSRB 单张交通标志图片识别",
        epilog="""示例:
  python predict.py test.png
  python predict.py data/GTSRB/Final_Test/Images/00000.ppm
  python predict.py my_photo.jpg --topk 10 --save"""
    )
    parser.add_argument("image", help="输入图片路径 支持 ppm jpg png bmp tiff 等")
    parser.add_argument("--topk", type=int, default=5, help="显示前 K 个预测结果 默认 5")
    parser.add_argument("--model", default=None, help="模型权重路径 默认 best_model.pth")
    parser.add_argument("--save", action="store_true", help="保存可视化结果图")
    parser.add_argument("--save_path", default="prediction_result.png",
                        help="结果图保存路径 默认 prediction_result.png")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_path = args.model or str(BASE_DIR / "best_model.pth")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            f"请先运行 train.py 训练模型，或通过 --model 指定权重路径"
        )

    print(f"加载模型: {model_path}")
    model = create_resnet34(num_classes=43, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"模型加载成功 参数量: {sum(p.numel() for p in model.parameters()):,}")

    pred_id, confidence, topk_ids, topk_confs, original_img = predict_image(
        model, args.image, device, topk=args.topk
    )

    result = display_results(
        pred_id, confidence, topk_ids, topk_confs,
        original_img, args.image, topk=args.topk
    )

    if args.save:
        save_visualization(result, original_img, args.save_path)

    if confidence < 0.5:
        print(f"模型对该结果置信度较低，建议人工复核")
    elif confidence < 0.7:
        print(f"模型对该结果置信度一般")


if __name__ == "__main__":
    main()

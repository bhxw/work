import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pathlib import Path

# 获取当前文件 (data_loader.py) 所在文件夹的绝对路径
# .parent 一次回到 src 文件夹，再 .parent 一次回到项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 构造数据路径（相对于项目根目录）
DATA_DIR = BASE_DIR / 'data' / 'GTSRB_training' / 'Final_Training' / 'Images'
# 2. 测试集图片路径
TEST_IMG_DIR = BASE_DIR / 'data' / 'GTSRB' / 'Final_Test' / 'Images'
# 3. 测试集标签 CSV 文件路径
#TEST_CSV_PATH = BASE_DIR / 'data' / 'GTSRB' / 'Final_Test' / 'Images' / 'GT-final_test.test.csv'
TEST_CSV_PATH = BASE_DIR / 'data' / 'GTSRB' / 'GT-final_test.csv'
print(f"数据路径: {DATA_DIR}")  # 方便调试，看路径是否正确

# ========== 1. 自定义训练集 Dataset ==========
class GTSRBTrainDataset(Dataset):
    """
    手动读取 GTSRB 训练集。
    目录结构为：
        root_dir/
            ├── 00000/
            │   ├── 00000_00000.ppm
            │   └── ...
            ├── 00001/
            └── ...
    文件夹名即类别标签（0~42）。
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 包含 00000~00042 子文件夹的路径
            transform (callable, optional): 图片预处理函数
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # 存放 (图片路径, 标签) 元组
        
        # 遍历43个文件夹 (00000 ~ 00042)
        for class_id in range(43):
            # 构造文件夹名，例如 '00000'，用 zfill 补齐5位
            folder_name = str(class_id).zfill(5)  # '00000' ~ '00042'
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                # 如果文件夹不存在（比如 GTSRB 只有 0~42，一般不会缺）
                continue
            # 遍历该文件夹下的所有文件
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.ppm'):
                    img_path = os.path.join(folder_path, file_name)
                    self.samples.append((img_path, class_id))
        
        print(f"训练集共加载 {len(self.samples)} 张图片")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        PyTorch 会根据索引调用这个函数，返回 (图片张量, 标签)
        """
        img_path, label = self.samples[idx]
        
        # 读取图片
        image = Image.open(img_path).convert('RGB')  # 转为RGB通道，确保3通道
        
        # 数据预处理
        if self.transform:
            image = self.transform(image)
        
        # 标签是整数，直接返回
        return image, label


# ========== 2. 定义预处理流水线 ==========
def get_train_transform():
    """
    训练集的数据增强 + 归一化
    """
    return transforms.Compose([
        # 1. 统一缩放到 224x224
        transforms.Resize((224, 224)),
        # 2. 随机旋转（±15度），增加泛化性
        transforms.RandomRotation(15),

        # 3. 转为张量
        transforms.ToTensor(),
        # 4. 归一化到 [-1, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """
    验证集/测试集：只做缩放和归一化，不做数据增强
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ========== 3. 批量读取并分割训练/验证集 ==========
def create_train_val_loaders(data_root, batch_size=64, val_ratio=0.2, num_workers=0):
    """
    创建训练集和验证集的 DataLoader
    
    Args:
        data_root: 训练集根目录（包含 00000~00042 文件夹）
        batch_size: 批次大小
        val_ratio: 验证集比例（0~1）
        num_workers: 并行读取线程数（Windows下建议设为0，避免多进程bug）
    
    Returns:
        train_loader, val_loader, dataset_size
    """
    # 先创建完整训练集（未分割）
    full_dataset = GTSRBTrainDataset(root_dir=data_root, 
                                     transform=get_train_transform())
    dataset_size = len(full_dataset)
    
    # 计算训练集和验证集大小
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    
    # 随机分割（设置随机种子保证可复现）
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 注意：random_split 返回的子集会继承原数据集的 transform，
    # 但验证集应该用 get_val_transform（无数据增强），
    # 所以需要单独替换 transform
    # 简便做法：在 random_split 之前，我们不设置 transform，
    # 而是在 __getitem__ 中根据模式选择？更规范的做法是重新封装。
    # 但为了代码简洁，我们可以先创建两个独立的 Dataset 对象，
    # 一个用 train_transform，一个用 val_transform。
    # 但这样会重复读取列表，稍微浪费内存。
    # 更好的方式：在 GTSRBTrainDataset 中增加一个参数 mode，
    # 但这里为了演示，我采用更直接的方法：
    
    # 我们重新创建两个 Dataset 对象，分别带不同的 transform
    # 但这就意味着要读两次文件列表，不过对于几万张图片，这不是问题。
    
    # 更优雅的方法：使用 Subset + 替换 transform 属性（高级用法）
    # 这里为清晰起见，直接重写两个 Dataset：
    
    # 重新加载全量数据（不指定 transform）
    full_dataset_no_transform = GTSRBTrainDataset(root_dir=data_root, transform=None)
    # 获取所有样本路径和标签列表
    all_samples = full_dataset_no_transform.samples
    all_labels = [s[1] for s in all_samples]  # 仅用于分层抽样（可选）
    
    # 使用 train_test_split 进行分层抽样（保持类别分布一致）
    from sklearn.model_selection import train_test_split
    train_indices, val_indices = train_test_split(
        range(len(all_samples)),
        test_size=val_ratio,
        stratify=all_labels,   # 根据标签分层
        random_state=42
    )
    
    # 创建子集 Dataset，并赋予不同的 transform
    # 这里我们利用 PyTorch 的 Subset 但需要重写其 transform 属性
    # 更简单：直接定义两个新类？算了，我们用最直接的方式：
    
    # 方法：构造两个新的 Dataset 实例，分别传入 transform
    train_dataset = GTSRBTrainDataset(root_dir=data_root, transform=get_train_transform())
    val_dataset = GTSRBTrainDataset(root_dir=data_root, transform=get_val_transform())
    
    # 但是这样它们包含了所有数据，我们需要用 Subset 截取索引
    # 但是上面已经读了两遍，效率低。因此推荐以下方案：
    
    # ************** 推荐方案：使用 Subset + 动态替换 transform **************
    # 以下展示官方推荐做法，代码稍微高级，但更高效：
    
    # 先创建不带 transform 的基础 Dataset
    base_dataset = GTSRBTrainDataset(root_dir=data_root, transform=None)
    # 获取全部索引
    indices = list(range(len(base_dataset)))
    # 按标签分层划分
    labels = [base_dataset.samples[i][1] for i in indices]
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, stratify=labels, random_state=42
    )
    
    # 创建 Subset 对象
    train_subset = torch.utils.data.Subset(base_dataset, train_idx)
    val_subset = torch.utils.data.Subset(base_dataset, val_idx)
    
    # 为子集动态绑定 transform
    # 注意：Subset 的 dataset 属性指向 base_dataset，我们修改 base_dataset.transform
    # 但这样会污染另一个子集，所以我们需要深拷贝或分别处理。
    # 更稳健：直接为 Subset 对象添加 transform 属性，并在 __getitem__ 中判断。
    # 为了简洁，我采用最直接的方法：创建两个 Dataset 实例，分别带不同 transform，然后用 Subset 截取。
    # 这样会读两次文件列表，但代码易懂。
    
    # ---------- 最终推荐（兼顾易懂与性能） ----------
    # 我直接写两个独立 Dataset，但共享文件列表？不，我选择用 lambda 或 partial。
    # 算了，对于课程作业，我们直接用最清晰的方式：
    
    # 我们只创建一次全量数据集，但设置 transform 为 None，
    # 然后在 __getitem__ 中根据标志应用不同的 transform？
    # 那就要在 Dataset 里传参，不优雅。
    
    # 最终我决定：使用两个独立的 Dataset 对象，分别带不同的 transform，
    # 然后用 Subset 截取它们的索引，但索引是对应各自的数据列表，所以需要一致。
    # 然而这样会重复加载，但几万张图片的路径列表，内存占用很小，没问题。
    
    # 为了节省篇幅，直接采用如下简易方式（被广泛使用）：
    # 先构建全量数据集，再分割，然后分别设置 transform（通过猴子补丁）
    full_dataset = GTSRBTrainDataset(root_dir=data_root, transform=None)
    labels = [s[1] for s in full_dataset.samples]
    train_idx, val_idx = train_test_split(range(len(full_dataset)), 
                                          test_size=val_ratio, 
                                          stratify=labels, 
                                          random_state=42)
    
    # 创建两个子集
    train_subset = torch.utils.data.Subset(full_dataset, train_idx)
    val_subset = torch.utils.data.Subset(full_dataset, val_idx)
    
    # 为 Subset 动态添加 transform 属性
    # 由于 Subset 的 __getitem__ 会调用 base_dataset 的 __getitem__，
    # 我们可以在 base_dataset 中增加一个 transform 属性，并允许修改。
    # 但两个子集需要不同的 transform，所以需要两个 base 实例。
    # 因此，还是用两个实例吧。
    
    # 好吧，为了让你能直接运行，我选择最清晰的方式：
    # 重新实例化两个 Dataset，分别带不同的 transform，然后再用 Subset 截取。
    # 这样内存占用微乎其微，因为只是路径列表的拷贝。
    
    # ----- 最终代码（可直接复制） -----
    train_dataset = GTSRBTrainDataset(root_dir=data_root, transform=get_train_transform())
    val_dataset = GTSRBTrainDataset(root_dir=data_root, transform=get_val_transform())
    
    # 但是这两个 dataset 的顺序完全相同吗？它们都遍历了文件夹，顺序一致。
    # 因此可以用相同的索引来分割（但必须保证一致）。
    # 我们基于 train_dataset 的样本列表进行分割，然后用相同索引作用于 val_dataset。
    labels = [s[1] for s in train_dataset.samples]
    train_idx, val_idx = train_test_split(range(len(train_dataset)), 
                                          test_size=val_ratio, 
                                          stratify=labels, 
                                          random_state=42)
    
    # 使用 Subset 截取
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    # 创建 DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, len(train_dataset)


# ========== 4. 测试集数据处理 ==========
import pandas as pd

class GTSRBTestDataset(Dataset):
    """
    读取 GTSRB 测试集。测试集图片在同一个文件夹内，标签从 CSV 读取。
    """
    def __init__(self, img_dir, csv_path, transform=None):

        self.img_dir = img_dir
        self.transform = transform
        
        # 读取 CSV，获取文件名和标签
        df = pd.read_csv(csv_path, sep=';')  # 注意分隔符是分号
        # CSV 列: 'Filename', 'Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'
        self.filenames = df['Filename'].values
        self.labels = df['ClassId'].values
        
        print(f"测试集共 {len(self.filenames)} 张图片")
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        label = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def create_test_loader(img_dir, csv_path, batch_size=64, num_workers=0):
    """
    创建测试集 DataLoader
    """
    transform = get_val_transform()  # 与验证集相同，无数据增强
    test_dataset = GTSRBTestDataset(img_dir, csv_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader



def save_all_images_as_numpy(dataset, save_path):

    import numpy as np
    data_list = []
    labels_list = []
    for img, label in dataset:  # 若 dataset 有 transform，则已经是张量
        data_list.append(img.numpy())  # 转为 numpy (C, H, W)
        labels_list.append(label)
    data_array = np.stack(data_list, axis=0)  # (N, C, H, W)
    labels_array = np.array(labels_list)
    np.save(save_path + '_data.npy', data_array)
    np.save(save_path + '_labels.npy', labels_array)
    print(f"保存完成，形状: {data_array.shape}")
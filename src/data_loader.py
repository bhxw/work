import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'GTSRB_training' / 'Final_Training' / 'Images'
TEST_IMG_DIR = BASE_DIR / 'data' / 'GTSRB' / 'Final_Test' / 'Images'
TEST_CSV_PATH = BASE_DIR / 'data' / 'GTSRB' / 'GT-final_test.csv'
print(f"数据路径: {DATA_DIR}")


class GTSRBTrainDataset(Dataset):
    """
    手动读取 GTSRB 训练集，文件夹名即类别标签 0~42
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_id in range(43):
            folder_name = str(class_id).zfill(5)
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.ppm'):
                    img_path = os.path.join(folder_path, file_name)
                    self.samples.append((img_path, class_id))

        print(f"训练集共加载 {len(self.samples)} 张图片")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def create_train_val_loaders(data_root, batch_size=64, val_ratio=0.2, num_workers=0):
    """
    创建训练集和验证集的 DataLoader，分层抽样保证类别分布一致
    """
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset

    train_dataset = GTSRBTrainDataset(root_dir=data_root, transform=get_train_transform())
    val_dataset = GTSRBTrainDataset(root_dir=data_root, transform=get_val_transform())

    labels = [s[1] for s in train_dataset.samples]
    train_idx, val_idx = train_test_split(
        range(len(train_dataset)),
        test_size=val_ratio,
        stratify=labels,
        random_state=42
    )

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset)


import pandas as pd


class GTSRBTestDataset(Dataset):
    """
    读取 GTSRB 测试集，从 CSV 获取图片名和标签
    """
    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_path, sep=';')
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
    transform = get_val_transform()
    test_dataset = GTSRBTestDataset(img_dir, csv_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader

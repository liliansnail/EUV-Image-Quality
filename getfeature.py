import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ==== 1. 自定义数据集 ====
class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path  # 返回图像和路径

# ==== 2. 加载模型 ====
from models import HyperNet  # 假设你模型写在 model.py 中
model = HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
model.load_state_dict(torch.load(r"D:\pythonProject3\hyperIQA-master\hyperIQA-master\pretrained\koniq_pretrained.pkl"))  # 加载权重
model.eval().cuda()

# ==== 3. 设置变换和 DataLoader ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolderDataset(r"D:\FY03IMG\database\simdegration", transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ==== 4. 提取特征向量 ====
all_features = []
all_paths = []

with torch.no_grad():
    for imgs, paths in tqdm(dataloader):
        imgs = imgs.cuda()
        out = model(imgs)
        vecs = out['target_in_vec'].cpu().numpy()  # shape: [B, D]
        all_features.append(vecs)
        all_paths.extend(paths)

all_features = np.concatenate(all_features, axis=0)  # shape: [N, D]

# 保存向量到文件
np.save(r"D:\FY03IMG\database\simdegration\HyperIqafeatures.npy", all_features)
with open('paths.txt', 'w') as f:
    for p in all_paths:
        f.write(p + '\n')

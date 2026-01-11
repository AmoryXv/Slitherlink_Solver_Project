# ... (Imports 和 SimpleDigitNet 类定义保持不变) ...
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class SimpleDigitNet(nn.Module):
    def __init__(self):
        super(SimpleDigitNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 5) 
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SyntheticDigitDataset(Dataset):
    def __init__(self, samples=5000):
        self.samples = samples
        self.img_size = 32
        # 混合字体库：既有细的也有粗的
        self.fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,       # 正常
            cv2.FONT_HERSHEY_PLAIN,         # 细
            cv2.FONT_HERSHEY_DUPLEX,        # 粗
            cv2.FONT_HERSHEY_COMPLEX        # 衬线
        ]

    def __len__(self): return self.samples

    def generate_image(self, label):
        img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255
        
        # === 空位生成 ===
        if label == 4: 
            # 只有 50% 的空位是脏的，另外 50% 是干净的
            if random.random() < 0.5:
                noise_level = random.randint(0, 30)
                # 修复了上一版的报错: 255-noise_level 可能等于 255
                low = 255 - noise_level
                img = np.random.randint(low, 256, (32, 32), dtype=np.uint8)
                
                # 偶尔加干扰
                if random.random() < 0.3:
                    cx, cy = random.randint(10, 22), random.randint(10, 22)
                    cv2.circle(img, (cx, cy), 1, 150, -1)
            return img

        # === 数字生成 ===
        font = random.choice(self.fonts)
        text = str(label)
        
        # 混合风格：有时细，有时粗
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            thickness = 1 # 细
        else:
            scale = random.uniform(1.1, 1.3)
            thickness = 2 # 粗

        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        
        # 绘制
        temp = np.zeros((64, 64), dtype=np.uint8)
        cv2.putText(temp, text, (20, 40), font, scale, (255,), thickness)
        
        coords = cv2.findNonZero(temp)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            digit_roi = temp[y:y+h, x:x+w]
            
            # 缩放至 22x22 (适中大小)
            target_size = 22.0
            scale_f = target_size / max(w, h)
            nw, nh = int(w * scale_f), int(h * scale_f)
            resized = cv2.resize(digit_roi, (nw, nh))
            
            canvas = np.zeros((32, 32), dtype=np.uint8)
            sx, sy = (32 - nw)//2, (32 - nh)//2
            canvas[sy:sy+nh, sx:sx+nw] = resized
            final = cv2.bitwise_not(canvas)
        else:
            final = img

        # 轻微噪点 (不要太重，以免破坏数字特征)
        if random.random() < 0.3:
            noise = np.random.randint(-5, 5, (32, 32))
            final = np.clip(final + noise, 0, 255).astype(np.uint8)
        
        return final

    def __getitem__(self, idx):
        # 保持 20% 是空位 (模拟真实情况，其实空位挺多的，但我们想多练练数字)
        if random.random() < 0.2: label = 4
        else: label = random.randint(0, 3)
            
        img = self.generate_image(label)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return img, label

def train():
    print("开始均衡训练 (Balanced Training)...")
    train_set = SyntheticDigitDataset(samples=8000)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleDigitNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5): # 5轮够了，模型很简单
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done.")

    torch.save(model.state_dict(), "digit_solver.pth")
    print("模型更新完毕！")

if __name__ == "__main__":
    train()
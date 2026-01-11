import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.cluster import KMeans
from collections import Counter
from vision_grid import GridExtractorV2 

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

class SlitherlinkOCR:
    def __init__(self, model_path="digit_solver.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleDigitNet().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(">>> 模型加载成功")
        except:
            print(">>> Warning: 模型未找到，将使用随机权重")
        self.model.eval()
        self.extractor = GridExtractorV2()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()
        self.current_cells = []

    def preprocess_cell_raw(self, img):
        if len(img.shape) == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: gray = img
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def smart_crop(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # 更严格的长宽比过滤，防止把横线当数字
        if w < 3 or h < 6: return None
        if w / float(h) > 2.5: return None # 过滤扁长的噪点

        digit_roi = thresh[y:y+h, x:x+w]
        canvas = np.zeros((32, 32), dtype=np.uint8)
        scale = 24.0 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0: return None
        
        resized = cv2.resize(digit_roi, (new_w, new_h))
        sx, sy = (32 - new_w)//2, (32 - new_h)//2
        canvas[sy:sy+new_h, sx:sx+new_w] = resized
        return cv2.bitwise_not(canvas)

    # === 核心升级：TTA 多重推理 ===
    def predict_with_tta(self, img_tensor):
        """
        Test-Time Augmentation (TTA)
        不仅仅预测一次，而是预测多次（原图、平移、缩放），取众数。
        """
        inputs = []
        inputs.append(img_tensor) # 1. 原图
        
        # 2. 向左平移 2px
        shifted = torch.roll(img_tensor, shifts=-2, dims=3) 
        shifted[:, :, :, -2:] = 1.0 # 填充白色
        inputs.append(shifted)
        
        # 3. 向右平移 2px
        shifted2 = torch.roll(img_tensor, shifts=2, dims=3)
        shifted2[:, :, :, :2] = 1.0
        inputs.append(shifted2)
        
        # 批量预测
        batch = torch.cat(inputs, dim=0) # (3, 1, 32, 32)
        with torch.no_grad():
            outputs = self.model(batch)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
        
        # 统计投票结果
        votes = preds.cpu().numpy().tolist()
        
        # 如果包含 0 和 空位(-1/4)，倾向于不轻易判 0，除非大家都说是 0
        final_label = Counter(votes).most_common(1)[0][0]
        
        # 如果 TTA 结果不一致（比如一次是0，两次是空），则偏向于认为是不确定的，判空
        # 这里简化：直接取众数
        
        # 取平均置信度
        avg_conf = confs.mean().item()
        
        return final_label, avg_conf

    def recognize_board_auto(self, image_path):
        warped, cells, rows, cols, debug_info = self.extractor.process_auto(image_path)
        self.current_cells = cells # 保存以供学习
        grid_matrix = np.zeros((rows, cols), dtype=int)

        ink_densities = []
        processed_cells = [] 

        # 1. 计算密度
        for r in range(rows):
            row_data = []
            for c in range(cols):
                thresh = self.preprocess_cell_raw(cells[r][c])
                pixel_count = cv2.countNonZero(thresh)
                ink_densities.append(pixel_count)
                row_data.append(thresh)
            processed_cells.append(row_data)

        # 2. 自适应阈值
        try:
            densities = np.array(ink_densities).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(densities)
            centers = sorted(kmeans.cluster_centers_.flatten())
            dynamic_threshold = (centers[0] + centers[1]) / 2
            if centers[1] - centers[0] < 50: dynamic_threshold = 50
        except: dynamic_threshold = 50 

        # 3. 识别
        for r in range(rows):
            for c in range(cols):
                thresh = processed_cells[r][c]
                pixel_count = ink_densities[r * cols + c]

                # 物理过滤
                if pixel_count < dynamic_threshold:
                    grid_matrix[r][c] = -1 
                    continue

                input_img = self.smart_crop(thresh)
                if input_img is None:
                    grid_matrix[r][c] = -1
                    continue
                
                # TTA 推理
                tensor = torch.from_numpy(input_img).float() / 255.0
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                
                label, conf = self.predict_with_tta(tensor)
                
                if label == 4: label = -1
                # 只有极低置信度才过滤，因为 TTA 已经做过一轮筛选了
                if label == 0 and conf < 0.5: label = -1
                    
                grid_matrix[r][c] = label

        return warped, grid_matrix, rows, cols, debug_info

    def learn_from_feedback(self, correct_matrix):
        # 保持之前的学习代码逻辑不变，这里为了节省篇幅简写，请保留你上一次的 learn_from_feedback 代码
        # (或者直接复制上一次的 learn_from_feedback 函数放在这里)
        # ---------------------------------------------------------
        print(">>> 正在从用户反馈中学习...")
        self.model.train()
        batch_imgs = []
        batch_labels = []
        rows = len(correct_matrix)
        cols = len(correct_matrix[0])
        learned_count = 0
        
        for r in range(rows):
            for c in range(cols):
                label = correct_matrix[r][c]
                if label not in [0, 1, 2, 3]: continue
                thresh = self.preprocess_cell_raw(self.current_cells[r][c])
                input_img = self.smart_crop(thresh)
                if input_img is not None:
                    img_tensor = torch.from_numpy(input_img).float() / 255.0
                    batch_imgs.append(img_tensor.unsqueeze(0))
                    batch_labels.append(label)
                    learned_count += 1
        
        if learned_count == 0: return 0
        inputs = torch.stack(batch_imgs).to(self.device)
        targets = torch.tensor(batch_labels).to(self.device)
        for _ in range(5):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        torch.save(self.model.state_dict(), self.model_path)
        self.model.eval()
        return learned_count
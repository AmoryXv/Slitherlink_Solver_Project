import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil
from sklearn.cluster import KMeans
from collections import Counter
from vision_grid import GridExtractorV2

# Tesseract 是可选依赖，不影响 CNN 主路径
try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

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

    # =========================================================
    #  预处理路径 A (原始)：equalizeHist + Otsu
    #  预处理路径 B (增强)：边框擦除 + 自适应阈值 + 形态学清洗
    # =========================================================

    def preprocess_cell_raw(self, img):
        """路径 A：原始预处理 (equalizeHist + Otsu)，对较清晰的图效果好"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        gray = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    def preprocess_cell_enhanced(self, img):
        """
        路径 B：增强版预处理。关键调参点：
        - 去掉 equalizeHist：避免放大网格圆点噪声
        - adaptive threshold (blockSize=19, C=7)：局部自适应
        - morphological opening (2x2 kernel)：去除 <3px 孤立噪点
        - 四边 12% 边框擦除：消除网格线碎片（从 15% 降到 12%，避免裁到数字）
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape[:2]

        # --- Step 1: 边框擦除 (12%) ---
        border_h = max(2, int(h * 0.12))
        border_w = max(2, int(w * 0.12))
        gray[:border_h, :] = 255
        gray[-border_h:, :] = 255
        gray[:, :border_w] = 255
        gray[:, -border_w:] = 255

        # --- Step 2: 自适应阈值 ---
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 19, 7
        )

        # --- Step 3: 形态学开运算 ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        return thresh

    def smart_crop(self, thresh):
        """
        智能裁剪：提取数字 ROI 并居中到 32x32 画布。
        增加面积占比过滤，排除「满屏噪点」情况。
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        if w < 3 or h < 6:
            return None
        if w / float(h) > 2.5:
            return None
        cell_area = thresh.shape[0] * thresh.shape[1]
        contour_area = cv2.contourArea(c)
        if cell_area > 0 and contour_area / cell_area < 0.02:
            return None

        digit_roi = thresh[y:y+h, x:x+w]
        canvas = np.zeros((32, 32), dtype=np.uint8)
        scale = 24.0 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0:
            return None

        resized = cv2.resize(digit_roi, (new_w, new_h))
        sx, sy = (32 - new_w) // 2, (32 - new_h) // 2
        canvas[sy:sy+new_h, sx:sx+new_w] = resized
        return cv2.bitwise_not(canvas)

    # =========================================================
    #  Tesseract 回退识别
    # =========================================================

    def recognize_cell_tesseract(self, cell_img):
        """
        使用 Tesseract OCR 识别单个数字 (0-3)。
        配置：--psm 10 (单字符模式), 白名单 0123。
        返回: int 标签 (0-3) 或 -1 (空位)。
        """
        if not _TESSERACT_AVAILABLE:
            return -1

        if len(cell_img.shape) == 3:
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_img.copy()

        h, w = gray.shape[:2]
        bh, bw = max(2, int(h * 0.12)), max(2, int(w * 0.12))
        gray[:bh, :] = 255
        gray[-bh:, :] = 255
        gray[:, :bw] = 255
        gray[:, -bw:] = 255

        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(
                binary,
                config='--psm 10 -c tessedit_char_whitelist=0123'
            ).strip()
            if text in ('0', '1', '2', '3'):
                return int(text)
        except Exception:
            pass
        return -1

    # =========================================================
    #  单路径 CNN 推理（不含 TTA，用于双路径对比）
    # =========================================================

    def _predict_single(self, input_img):
        """对单张 32x32 图像做一次前向推理，返回 (label, confidence)"""
        tensor = torch.from_numpy(input_img).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        label = pred.item()
        if label == 4:
            label = -1
        return label, conf.item()

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
        final_label = Counter(votes).most_common(1)[0][0]
        avg_conf = confs.mean().item()
        
        return final_label, avg_conf

    def recognize_board_auto(self, image_path, debug_dump=False, use_tesseract=False):
        """
        自动识别棋盘（双路径预处理 + 置信度投票）。
        
        对每个含数字的格子，同时运行两种预处理：
          路径 A (原始): equalizeHist + Otsu
          路径 B (增强): 边框擦除 + adaptive threshold + morphological opening
        取两者中置信度更高的结果。这样不同风格的图片都能获得最佳识别效果。
        
        参数:
            image_path:    图片路径
            debug_dump:    True 时将中间裁剪图保存到 debug_crops/
            use_tesseract: True 时使用 Tesseract 替代 CNN
        """
        warped, cells, rows, cols, debug_info = self.extractor.process_auto(image_path)
        self.current_cells = cells
        grid_matrix = np.zeros((rows, cols), dtype=int)

        # --- debug_dump 初始化 ---
        dump_dir = None
        if debug_dump:
            dump_dir = os.path.join(os.path.dirname(os.path.abspath(image_path)), 'debug_crops')
            if os.path.exists(dump_dir):
                shutil.rmtree(dump_dir)
            os.makedirs(dump_dir, exist_ok=True)
            print(f"[Debug] 中间裁剪图将保存到: {dump_dir}")

        # --- Tesseract 检查 ---
        if use_tesseract and not _TESSERACT_AVAILABLE:
            print(">>> Warning: pytesseract 未安装，回退到 CNN 模式")
            use_tesseract = False

        # -------------------------------------------------------
        #  Phase 1: 双路径预处理 + 密度计算
        # -------------------------------------------------------
        raw_cells = []      # 路径 A 的二值化结果
        enhanced_cells = []  # 路径 B 的二值化结果
        ink_densities = []   # 用路径 A 计算密度（兼容旧行为）

        for r in range(rows):
            raw_row, enh_row = [], []
            for c in range(cols):
                cell = cells[r][c]
                # 路径 A：原始预处理
                thresh_a = self.preprocess_cell_raw(cell)
                # 路径 B：增强预处理
                thresh_b = self.preprocess_cell_enhanced(cell)

                pixel_count = cv2.countNonZero(thresh_a)
                ink_densities.append(pixel_count)
                raw_row.append(thresh_a)
                enh_row.append(thresh_b)

                if dump_dir:
                    if len(cell.shape) == 3:
                        raw_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    else:
                        raw_gray = cell
                    cv2.imwrite(os.path.join(dump_dir, f'r{r}_c{c}_raw.png'), raw_gray)
                    cv2.imwrite(os.path.join(dump_dir, f'r{r}_c{c}_threshA.png'), thresh_a)
                    cv2.imwrite(os.path.join(dump_dir, f'r{r}_c{c}_threshB.png'), thresh_b)

            raw_cells.append(raw_row)
            enhanced_cells.append(enh_row)

        # -------------------------------------------------------
        #  Phase 2: 自适应墨水密度阈值 (KMeans)
        # -------------------------------------------------------
        try:
            densities = np.array(ink_densities).reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(densities)
            centers = sorted(kmeans.cluster_centers_.flatten())
            dynamic_threshold = (centers[0] + centers[1]) / 2
            if centers[1] - centers[0] < 50:
                dynamic_threshold = 50
        except Exception:
            dynamic_threshold = 50

        # -------------------------------------------------------
        #  Phase 3: 识别（双路径投票）
        # -------------------------------------------------------
        for r in range(rows):
            for c in range(cols):
                pixel_count = ink_densities[r * cols + c]

                # 物理过滤：密度太低 → 空位
                if pixel_count < dynamic_threshold:
                    grid_matrix[r][c] = -1
                    continue

                # ----- Tesseract 路径 -----
                if use_tesseract:
                    grid_matrix[r][c] = self.recognize_cell_tesseract(cells[r][c])
                    continue

                # ----- 双路径 CNN 投票 -----
                thresh_a = raw_cells[r][c]
                thresh_b = enhanced_cells[r][c]

                crop_a = self.smart_crop(thresh_a)
                crop_b = self.smart_crop(thresh_b)

                label_a, conf_a = -1, 0.0
                label_b, conf_b = -1, 0.0

                if crop_a is not None:
                    label_a, conf_a = self._predict_single(crop_a)
                if crop_b is not None:
                    label_b, conf_b = self._predict_single(crop_b)

                # 投票策略：取置信度更高的结果
                if label_a == label_b:
                    # 两路一致，直接采纳
                    best_label, best_conf = label_a, max(conf_a, conf_b)
                elif conf_a >= conf_b:
                    best_label, best_conf = label_a, conf_a
                else:
                    best_label, best_conf = label_b, conf_b

                # 极低置信度的 0 过滤
                if best_label == 0 and best_conf < 0.5:
                    best_label = -1

                grid_matrix[r][c] = best_label

                if dump_dir:
                    if crop_a is not None:
                        cv2.imwrite(os.path.join(dump_dir, f'r{r}_c{c}_cropA.png'), crop_a)
                    if crop_b is not None:
                        cv2.imwrite(os.path.join(dump_dir, f'r{r}_c{c}_cropB.png'), crop_b)

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
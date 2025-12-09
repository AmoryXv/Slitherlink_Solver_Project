import cv2
import numpy as np
import os
import pickle

class DigitClassifier:
    def __init__(self, model_path="ocr_brain.pkl"):
        self.model_path = model_path
        self.templates = {}
        self.learned_count = 0
        self.threshold = 500000 
        
        # 启动时，尝试从硬盘加载记忆
        self.load_model()

    def load_model(self):
        """从硬盘读取记忆"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.templates = data.get("templates", {})
                    self.learned_count = data.get("count", 0)
                print(f"[OCR] 成功加载记忆库，包含 {self.learned_count} 个样本。")
            except Exception as e:
                print(f"[OCR] 加载记忆失败: {e}")
        else:
            print("[OCR] 未找到记忆文件，初始化新大脑。")

    def save_model(self):
        """把记忆保存到硬盘"""
        data = {
            "templates": self.templates,
            "count": self.learned_count
        }
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(data, f)
            print("[OCR] 记忆已保存到硬盘。")
        except Exception as e:
            print(f"[OCR] 保存记忆失败: {e}")

    def preprocess(self, roi):
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
        resized = cv2.resize(binary, (30, 30))
        return resized

    def classify(self, roi_img):
        target = self.preprocess(roi_img)
        best_label = None
        min_diff = float('inf')

        for label, examples in self.templates.items():
            for template in examples:
                diff = np.sum((target.astype("float") - template.astype("float")) ** 2)
                if diff < min_diff:
                    min_diff = diff
                    best_label = label

        if best_label is not None and min_diff < self.threshold:
            return best_label, True
        else:
            return None, False

    def learn(self, roi_img, label):
        target = self.preprocess(roi_img)
        if label not in self.templates:
            self.templates[label] = []
        self.templates[label].append(target)
        self.learned_count += 1
        
        print(f"[OCR] 已学习数字 '{label}' (样本 #{self.learned_count})")
        # [修改] 注释掉这一行！不要每次学完都存盘，太慢了！
        # self.save_model()
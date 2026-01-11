import cv2
import numpy as np

class GridExtractorV2:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 5)
        return thresh

    def analyze_grid_structure(self, thresh_img):
        """
        [V7 中位数强过滤版]
        使用 Median 统计学方法，只有点数接近中位数的线才会被保留。
        彻底解决 "两个0形成一条线" 的问题。
        """
        contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for c in contours:
            area = cv2.contourArea(c)
            # 稍微调宽一点面积范围，防止大图的点被漏
            if 2 < area < 1000: 
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.4:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dots.append([cx, cy])
        
        if len(dots) < 16:
            raise ValueError(f"检测到的点太少 ({len(dots)})，无法构成网格。")

        dots = np.array(dots)
        img_w = np.max(dots[:, 0]) - np.min(dots[:, 0])
        tolerance = max(8, img_w / 60)

        def cluster_1d(coords):
            if len(coords) == 0: return []
            coords = sorted(coords)
            clusters = []
            current_group = [coords[0]]
            for x in coords[1:]:
                if abs(x - np.mean(current_group)) < tolerance:
                    current_group.append(x)
                else:
                    clusters.append(np.mean(current_group))
                    current_group = [x]
            clusters.append(np.mean(current_group))
            return np.array(clusters)

        candidate_v = cluster_1d(dots[:, 0])
        candidate_h = cluster_1d(dots[:, 1])

        # === 核心修改：中位数强过滤 ===
        def filter_strict(lines, axis_idx):
            # 1. 统计每条线上的点数
            counts = []
            for line_pos in lines:
                cnt = 0
                for dot in dots:
                    if abs(dot[axis_idx] - line_pos) < tolerance:
                        cnt += 1
                counts.append(cnt)
            
            if not counts: return np.array([])
            
            # 2. 计算中位数 (Median)
            # 例如：大部分列有 8 个点，那么 median 就是 8
            # 哪怕有两列是 2 个点 (噪点)，median 依然稳坐 8
            median_count = np.median([c for c in counts if c > 1]) # 忽略只有1个点的纯噪点
            
            # 3. 设定严格阈值
            # 规则：必须至少达到 (中位数 - 1) 个点
            # 如果 Median=8，则至少要有 7 个点。
            # 这样，2 个 0 组成的 4 个点绝对会被杀掉。
            threshold = max(3, median_count - 1)
            
            valid_lines = []
            for i, line_pos in enumerate(lines):
                if counts[i] >= threshold:
                    valid_lines.append(line_pos)
            
            return np.array(valid_lines)

        v_lines = filter_strict(candidate_v, 0)
        h_lines = filter_strict(candidate_h, 1)

        if len(v_lines) < 2 or len(h_lines) < 2:
            raise ValueError("有效网格线不足，无法识别结构")

        min_x, max_x = v_lines[0], v_lines[-1]
        min_y, max_y = h_lines[0], h_lines[-1]

        corners = np.array([
            [min_x, min_y], [max_x, min_y], 
            [max_x, max_y], [min_x, max_y]
        ], dtype="float32")

        debug_info = {'v_lines': v_lines, 'h_lines': h_lines, 'dots': dots}
        return corners, len(h_lines)-1, len(v_lines)-1, debug_info

    def four_point_transform(self, image, rect):
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def slice_grid(self, warped_img, rows, cols):
        img_h, img_w = warped_img.shape[:2]
        cell_h = img_h / rows
        cell_w = img_w / cols
        cells = []
        # 保持 8% 的边距，不切数字
        margin_h = int(cell_h * 0.08)
        margin_w = int(cell_w * 0.08)

        for r in range(rows):
            row_cells = []
            for c in range(cols):
                y1 = int(r * cell_h + margin_h)
                y2 = int((r + 1) * cell_h - margin_h)
                x1 = int(c * cell_w + margin_w)
                x2 = int((c + 1) * cell_w - margin_w)
                y1, y2 = max(0, y1), min(img_h, y2)
                x1, x2 = max(0, x1), min(img_w, x2)
                row_cells.append(warped_img[y1:y2, x1:x2])
            cells.append(row_cells)
        return cells

    def process_auto(self, image_path):
        image = cv2.imread(image_path)
        if image is None: raise ValueError("无法读取图片")
        thresh = self.preprocess_image(image)
        corners, rows, cols, debug_info = self.analyze_grid_structure(thresh)
        print(f"[AutoDetect] 结构: {rows} x {cols}")
        warped = self.four_point_transform(image, corners)
        cells = self.slice_grid(warped, rows, cols)
        return warped, cells, rows, cols, debug_info
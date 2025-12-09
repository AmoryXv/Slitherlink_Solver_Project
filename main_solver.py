import cv2
import numpy as np
import time
from puzzle_model import SlitherlinkPuzzle
from digit_classifier import DigitClassifier

def solve_image_puzzle_smart(image_path):
    print(f"\n>>> [Step 1] 视觉分析: {image_path}")
    img = cv2.imread(image_path)
    if img is None: 
        print("错误：找不到图片")
        return

    # --- 1. 图像处理 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # --- 2. 提取轮廓 ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    digit_boxes = []
    
    min_dot_area = 5
    max_dot_area = 150

    for c in contours:
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if min_dot_area < area < max_dot_area:
            dots.append((cx, cy))
        elif area >= max_dot_area:
            x, y, w, h = cv2.boundingRect(c)
            center_x = x + w // 2
            center_y = y + h // 2
            digit_boxes.append({'cx': center_x, 'cy': center_y, 'rect': (x, y, w, h)})

    # --- 3. 构建精准网格 ---
    if len(dots) < 4: 
        print("未检测到足够的点阵")
        return

    # 点排序：先按 Y 模糊分行，再按 X 排序
    dots.sort(key=lambda p: p[1]) 
    rows = []
    current_row = [dots[0]]
    for i in range(1, len(dots)):
        p = dots[i]
        if abs(p[1] - current_row[-1][1]) < 15: current_row.append(p)
        else:
            current_row.sort(key=lambda p: p[0])
            rows.append(current_row)
            current_row = [p]
    current_row.sort(key=lambda p: p[0])
    rows.append(current_row)

    num_rows, num_cols = len(rows), len(rows[0])
    print(f"  -> 精准锁定结构: {num_rows}行 x {num_cols}列 点阵")
    
    # --- 4. 智能识别 ---
    print("\n>>> [Step 2] 智能字符识别 (自学习模式)")
    puzzle_h, puzzle_w = num_rows - 1, num_cols - 1
    clues_matrix = [[-1 for _ in range(puzzle_w)] for _ in range(puzzle_h)]
    ocr_engine = DigitClassifier()

    for r in range(puzzle_h):
        for c in range(puzzle_w):
            p1, p2 = rows[r][c], rows[r][c+1]
            p3, p4 = rows[r+1][c], rows[r+1][c+1]
            
            cell_min_x = min(p1[0], p3[0]); cell_max_x = max(p2[0], p4[0])
            cell_min_y = min(p1[1], p2[1]); cell_max_y = max(p3[1], p4[1])
            
            found_digit = None
            for dbox in digit_boxes:
                dcx, dcy = dbox['cx'], dbox['cy']
                margin = 5
                if (cell_min_x - margin < dcx < cell_max_x + margin) and \
                   (cell_min_y - margin < dcy < cell_max_y + margin):
                    found_digit = dbox
                    break
            
            if found_digit:
                x, y, w, h = found_digit['rect']
                roi = img[y:y+h, x:x+w]
                label, is_confident = ocr_engine.classify(roi)
                
                final_val = -1
                if is_confident:
                    print(f"  (自动) ({r},{c}) -> {label}")
                    final_val = int(label)
                else:
                    roi_display = cv2.resize(roi, (100, 100), interpolation=cv2.INTER_NEAREST)
                    win_name = f"Check ({r},{c})"
                    cv2.imshow(win_name, roi_display)
                    cv2.moveWindow(win_name, 400, 300)
                    cv2.waitKey(100)
                    while True:
                        try:
                            user_input = input(f"  [人类] 格子({r},{c}) 是几? > ")
                            if user_input in ['0', '1', '2', '3']:
                                ocr_engine.learn(roi, user_input)
                                final_val = int(user_input)
                                break
                        except: pass
                    cv2.destroyAllWindows()
                clues_matrix[r][c] = final_val

    # --- 5. 求解 ---
    print("\n--- 最终识别矩阵 ---")
    for row in clues_matrix:
        print(" ".join([str(x) if x != -1 else "." for x in row]))
    print("--------------------")
    
    print("\n>>> [Step 3] 启动工业级求解器")
    solver = SlitherlinkPuzzle(puzzle_h, puzzle_w, clues_matrix)
    
    # [关键步骤] 先跑预处理
    solver.apply_basic_rules()
    
    start_time = time.time()
    if solver.solve_backtracking():
        end_time = time.time()
        print(f"✅ 求解成功！耗时: {(end_time - start_time)*1000:.2f} ms")
        solver.print_board_fancy()
    else:
        print("❌ 无解")

if __name__ == "__main__":
    solve_image_puzzle_smart("test_puzzle.png")
import cv2
import numpy as np

def detect_grid_structure(image_path):
    print(f"正在分析图片结构: {image_path} ...")
    img = cv2.imread(image_path)
    
    # --- 1. 预处理 (和之前一样) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # --- 2. 找轮廓 ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 3. 智能分类：是点还是数字？ ---
    dots = []    # 存放点
    digits = []  # 存放数字
    
    # 我们可以统计一下所有轮廓的面积，找出规律
    # 但对于 Slitherlink，点通常非常小
    min_dot_area = 5
    max_dot_area = 150  # 经验值：点一般不会超过这个大小

    visualization = img.copy()

    for c in contours:
        area = cv2.contourArea(c)
        
        # 获取轮廓的中心坐标 (cx, cy)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue

        if min_dot_area < area < max_dot_area:
            # 认为是点 -> 画蓝色圆圈
            dots.append((cx, cy))
            cv2.circle(visualization, (cx, cy), 5, (255, 0, 0), -1) # Blue
        elif area >= max_dot_area:
            # 认为是数字 -> 画红色方框
            digits.append((cx, cy))
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red

    # --- 4. 推算网格尺寸 ---
    # 既然我们找到了所有的点，我们就可以数数看有多少行、多少列
    # 这是一个简单的估算逻辑：
    if len(dots) > 0:
        # 把点按 Y 坐标排序，相近的算同一行
        dots_y_sorted = sorted(dots, key=lambda p: p[1])
        
        # 简单的聚类算法：如果两个点的 Y 坐标差值很小，就算同一行
        rows = 1
        current_y = dots_y_sorted[0][1]
        
        for p in dots_y_sorted[1:]:
            if abs(p[1] - current_y) > 20: # 假设行高至少大于20像素
                rows += 1
                current_y = p[1]
        
        # 总点数 / 行数 = 列数
        cols = round(len(dots) / rows)
        
        print(f"\n✅ 分析完成！")
        print(f"  -> 找到了 {len(dots)} 个定位点 (蓝色)")
        print(f"  -> 找到了 {len(digits)} 个数字线索 (红色)")
        print(f"  -> 推测网格结构: {rows} 行 x {cols} 列 的点阵")
        print(f"  -> 对应谜题大小: {rows-1} x {cols-1} 格")
        
        # 把推测结果写在图上
        cv2.putText(visualization, f"Grid: {rows-1}x{cols-1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Structure Analysis", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 运行
detect_grid_structure("test_puzzle.png") # 注意用你的原图文件名
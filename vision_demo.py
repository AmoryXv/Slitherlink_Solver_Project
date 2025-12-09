import cv2
import numpy as np

def process_image(image_path):
    print(f"正在读取图片: {image_path} ...")
    
    # 1. 读取图片
    # cv2.imread 默认读进来是 BGR 颜色模式
    img = cv2.imread(image_path)
    
    if img is None:
        print("❌ 错误：找不到图片！请确认文件名是否正确，或者图片是否在项目文件夹里。")
        return

    # 2. 调整大小 (Resize)
    # 为了处理速度快，如果图片太大，我们把它缩小一点
    # 这一步在工程上很重要，防止高清大图把算法跑崩
    height, width = img.shape[:2]
    max_height = 800
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        img = cv2.resize(img, (new_width, max_height))
        print(f"图片太大，已缩放至: {new_width}x{max_height}")

    # 3. 灰度化 (Grayscale)
    # 颜色对于识别网格没用，扔掉颜色信息能减少计算量
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. 高斯模糊 (Gaussian Blur)
    # 这一步是为了“磨皮”，把纸张上的噪点、污渍去掉，只保留主要线条
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 5. 自适应二值化 (Adaptive Thresholding)
    # 这是识别网格的神器！它会自动根据局部亮度把图片变成只有“黑与白”
    # 任何比周围暗的东西（比如墨水线）都会变成白色（或者黑色，看设置）
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, # INV表示反转：线条变白，背景变黑
        11, # 邻域大小
        2   # 常数
    )

    # --- 显示结果 ---
    # OpenCV 会弹出窗口显示图片
    print(">>> 已打开图片窗口。按键盘任意键（如空格）退出...")
    
    cv2.imshow("1. Original", img)
    cv2.imshow("2. Grayscale", gray)
    cv2.imshow("3. Binary (Lines Detected)", binary)
    # ... (上半部分读取图片、二值化的代码保持不变) ...

    print(">>> 正在寻找拼图区域 (基于内容检测)...")

    # 6. 寻找所有轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. 过滤噪点
    # 我们只保留那些稍微大一点的轮廓（去掉扫描仪造成的微小杂色）
    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) > 10:  # 面积大于10像素才算有效内容
            valid_contours.append(c)

    if len(valid_contours) > 0:
        # 8. 核心魔法：把所有分散的轮廓（点、数字）合并成一个大整体
        all_points = np.concatenate(valid_contours)
        
        # 9. 计算这个大整体的边界框 (Bounding Rect)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # 稍微给框加一点“内边距” (Padding)，别贴得太紧
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + padding * 2)
        h = min(img.shape[0] - y, h + padding * 2)

        print(f"✅ 找到了！拼图区域在: x={x}, y={y}, w={w}, h={h}")

        # 10. 绘制结果
        # 画出绿色的框
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # 为了让你看清它是怎么判定的，我们把所有识别到的“小墨水点”也用红色标出来
        cv2.drawContours(img, valid_contours, -1, (0, 0, 255), 1)

        # 11. 裁剪出拼图区域 (Crop)
        # 这就是这一步的最终目的：把无关的白边切掉，只留题目
        puzzle_crop = img[y:y+h, x:x+w]
        cv2.imshow("5. Final Crop", puzzle_crop)

        cv2.imshow("4. Puzzle Detected", img)
    else:
        print("⚠️ 警告：这张图是一张白纸吗？什么都没找到。")

    # --- 等待代码 ---
    print(">>> 再次按键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # # 等待按键，0 表示无限等待
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# --- 运行主程序 ---
# 请确保你的文件名是 test_puzzle.jpg，或者是 png
process_image("test_puzzle.png")


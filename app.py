import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import pandas as pd 

from puzzle_model import SlitherlinkPuzzle
from digit_classifier import DigitClassifier

# --- 1. 页面配置 ---
st.set_page_config(page_title="Slitherlink AI Solver", layout="wide")
st.title("🧩 Slitherlink AI Solver (数回智能解题)")

# --- 2. 初始化 ---
def init_session():
    # OCR 引擎持久化
    if 'ocr_engine' not in st.session_state:
        st.session_state['ocr_engine'] = DigitClassifier()
    
    # 核心数据存储
    if 'puzzle_data' not in st.session_state:
        st.session_state['puzzle_data'] = None
    if 'cell_rois' not in st.session_state:
        st.session_state['cell_rois'] = {}
    if 'current_file_id' not in st.session_state:
        st.session_state['current_file_id'] = None

init_session()

# --- 3. 侧边栏 ---
with st.sidebar:
    st.header("🛠️ 面板")
    uploaded_file = st.file_uploader("上传题目", type=['png', 'jpg', 'jpeg'], key="uploader")
    
    # 检测新图片
    file_id = uploaded_file.file_id if uploaded_file else None
    if file_id != st.session_state['current_file_id']:
        st.session_state['puzzle_data'] = None
        st.session_state['cell_rois'] = {}
        st.session_state['current_file_id'] = file_id

    if st.button("🗑️ 重置所有记忆"):
        if os.path.exists("ocr_brain.pkl"):
            os.remove("ocr_brain.pkl")
        st.session_state['ocr_engine'] = DigitClassifier()
        st.success("记忆已清空！")

# --- 4. 主逻辑 ---
def process_ui(image_path):
    img = cv2.imread(image_path)
    if img is None: return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="原始题目", channels="BGR", use_container_width=True)

    # === 图像处理 ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots, digit_boxes = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if 5 < area < 150:
            M = cv2.moments(c)
            if M["m00"] != 0:
                dots.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))
        elif area >= 150:
            x,y,w,h = cv2.boundingRect(c)
            digit_boxes.append({'cx': x+w//2, 'cy': y+h//2, 'rect':(x,y,w,h)})

    if len(dots) < 4:
        st.error("无法识别结构"); return

    # === 构建坐标系 ===
    dots.sort(key=lambda p: p[1])
    rows = []
    current_row = [dots[0]]
    for i in range(1, len(dots)):
        if abs(dots[i][1] - current_row[-1][1]) < 15: current_row.append(dots[i])
        else:
            current_row.sort(key=lambda p: p[0])
            rows.append(current_row)
            current_row = [dots[i]]
    current_row.sort(key=lambda p: p[0])
    rows.append(current_row)
    
    ph, pw = len(rows)-1, len(rows[0])-1
    
    # === 首次识别 (仅在新图时运行) ===
    if st.session_state['puzzle_data'] is None:
        init_mat = [[-1]*pw for _ in range(ph)]
        new_rois = {}
        ocr = st.session_state['ocr_engine']
        
        for r in range(ph):
            for c in range(pw):
                p1, p4 = rows[r][c], rows[r+1][c+1]
                min_x, max_x = min(p1[0], rows[r+1][c][0]), max(rows[r][c+1][0], p4[0])
                min_y, max_y = min(p1[1], rows[r][c+1][1]), max(rows[r+1][c][1], p4[1])
                
                found = None
                for db in digit_boxes:
                    if (min_x-5 < db['cx'] < max_x+5) and (min_y-5 < db['cy'] < max_y+5):
                        found = db; break
                
                if found:
                    x,y,w,h = found['rect']
                    roi = img[y:y+h, x:x+w]
                    new_rois[(r,c)] = roi
                    label, conf = ocr.classify(roi)
                    if conf: init_mat[r][c] = int(label)
        
        st.session_state['puzzle_data'] = init_mat
        st.session_state['cell_rois'] = new_rois

    # === 右侧交互区 ===
    with col2:
        st.subheader("识别结果与修正")
        st.info("👇 请直接修改下方表格，按回车生效")

        # [修复核心] 
        # 1. 把 list 转成 DataFrame 给编辑器显示
        # 2. 直接获取编辑器的返回值 (result_df)，而不是用回调
        current_list = st.session_state['puzzle_data']
        df_display = pd.DataFrame(current_list)
        df_display.columns = [i for i in range(pw)] # 强制列名

        result_df = st.data_editor(
            df_display,
            key="matrix_editor",
            use_container_width=True,
            height=300,
            hide_index=True 
        )

        # [关键同步]
        # 每次页面刷新，把编辑器最新的结果存回 puzzle_data
        # 这样数据流是单向的：Editor -> Session -> Solver，绝对不会类型错乱
        try:
            # 暴力清洗数据：空值填-1，非数字转-1，最后转int
            clean_df = result_df.apply(pd.to_numeric, errors='coerce').fillna(-1).astype(int)
            st.session_state['puzzle_data'] = clean_df.values.tolist()
        except Exception as e:
            st.error(f"数据格式错误: {e}")

        st.divider()

        # === 求解按钮 ===
        if st.button("🚀 学习并求解", type="primary", use_container_width=True):
            current_data = st.session_state['puzzle_data'] # 使用刚刚同步的最新数据
            rois = st.session_state['cell_rois']
            ocr = st.session_state['ocr_engine']
            learned = 0
            
            # 1. 隐式学习
            for r in range(ph):
                for c in range(pw):
                    val = current_data[r][c]
                    if (r,c) in rois and val in [0,1,2,3]:
                        ocr.learn(rois[(r,c)], str(val))
                        learned += 1
            
            # 批量存盘
            if learned > 0: 
                try:
                    ocr.save_model() # 确保 digit_classifier.py 有这个方法
                    st.toast(f"已保存 {learned} 个新字形！", icon="💾")
                except:
                    st.toast(f"学习了 {learned} 个新字形 (内存模式)", icon="🧠")

            # 2. 求解
            solver = SlitherlinkPuzzle(ph, pw, current_data)
            
            try:
                solver.apply_basic_rules()
                if solver.solve_backtracking():
                    st.success("✅ 求解成功！")
                    
                    # 画图
                    res_img = img.copy()
                    for r in range(ph+1):
                        for c in range(pw):
                            if solver.h_edges[r][c]==1:
                                cv2.line(res_img, rows[r][c], rows[r][c+1], (0,0,255), 3)
                            elif solver.h_edges[r][c]==2:
                                pt = ((rows[r][c][0]+rows[r][c+1][0])//2, (rows[r][c][1]+rows[r][c+1][1])//2)
                                cv2.drawMarker(res_img, pt, (200,200,200), cv2.MARKER_CROSS, 8, 1)
                    for r in range(ph):
                        for c in range(pw+1):
                            if solver.v_edges[r][c]==1:
                                cv2.line(res_img, rows[r][c], rows[r+1][c], (0,0,255), 3)
                            elif solver.v_edges[r][c]==2:
                                pt = ((rows[r][c][0]+rows[r+1][c][0])//2, (rows[r][c][1]+rows[r+1][c][1])//2)
                                cv2.drawMarker(res_img, pt, (200,200,200), cv2.MARKER_CROSS, 8, 1)
                    
                    st.image(res_img, channels="BGR", use_container_width=True)
                else:
                    st.error("无解，请检查是否有数字填错（例如 3 填成了 2）")
            except Exception as e:
                st.error(f"求解器错误: {e}")

if __name__ == "__main__":
    if 'uploader' in st.session_state and st.session_state.uploader:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            f.write(st.session_state.uploader.getbuffer())
            tmp = f.name
        process_ui(tmp)
        os.remove(tmp)
    else:
        st.info("👈 请在左侧上传图片")
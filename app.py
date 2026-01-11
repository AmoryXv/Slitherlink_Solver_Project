import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import pandas as pd
from ocr_engine import SlitherlinkOCR
from puzzle_model import SlitherlinkPuzzle

st.set_page_config(page_title="Slitherlink Auto Solver V3", layout="wide")
st.title("🧩 Slitherlink Auto Solver Pro")

if 'ocr' not in st.session_state:
    st.session_state['ocr'] = SlitherlinkOCR()
if 'matrix' not in st.session_state:
    st.session_state['matrix'] = None
if 'debug_info' not in st.session_state:
    st.session_state['debug_info'] = None

with st.sidebar:
    st.header("操作面板")
    uploaded_file = st.file_uploader("上传题目", type=['png', 'jpg'])
    st.divider()
    
    # 增加手动修正选项
    use_manual = st.checkbox("手动指定规格 (如果自动检测出错)")
    if use_manual:
        manual_rows = st.number_input("行数", 3, 30, 6)
        manual_cols = st.number_input("列数", 3, 30, 6)

def solve_it(matrix, warped_img):
    solver = SlitherlinkPuzzle(len(matrix), len(matrix[0]), matrix)
    solver.apply_basic_rules()
    if solver.solve_backtracking():
        # 画图
        res_img = warped_img.copy()
        h, w = res_img.shape[:2]
        ch, cw = h / solver.height, w / solver.width 
        
        for r in range(solver.height + 1):
            for c in range(solver.width):
                if solver.h_edges[r][c] == 1:
                    cv2.line(res_img, (int(c*cw), int(r*ch)), (int((c+1)*cw), int(r*ch)), (0,0,255), 4)
                elif solver.h_edges[r][c] == 2:
                    cx = int(c*cw + cw/2)
                    cv2.drawMarker(res_img, (cx, int(r*ch)), (200,200,200), cv2.MARKER_CROSS, 10, 2)
        
        for r in range(solver.height):
            for c in range(solver.width + 1):
                if solver.v_edges[r][c] == 1:
                    cv2.line(res_img, (int(c*cw), int(r*ch)), (int(c*cw), int((r+1)*ch)), (0,0,255), 4)
                elif solver.v_edges[r][c] == 2:
                    cy = int(r*ch + ch/2)
                    cv2.drawMarker(res_img, (int(c*cw), cy), (200,200,200), cv2.MARKER_CROSS, 10, 2)
        
        st.success("✅ 求解成功")
        st.image(res_img, channels="BGR", caption="Solution")
    else:
        st.error("❌ 无解 (请检查数字是否正确)")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
        f.write(uploaded_file.getbuffer())
        tmp_path = f.name

    if st.session_state.get('last_file') != uploaded_file.name:
        with st.spinner("🔍 正在扫描网格点..."):
            try:
                # 自动识别
                warped, matrix, rows, cols, debug_info = st.session_state['ocr'].recognize_board_auto(tmp_path)
                
                st.session_state['matrix'] = matrix
                st.session_state['warped'] = warped
                st.session_state['rows'] = rows
                st.session_state['cols'] = cols
                st.session_state['debug_info'] = debug_info
                st.session_state['last_file'] = uploaded_file.name
            except Exception as e:
                st.error(f"识别出错: {e}")

    # 显示结果
    if st.session_state['matrix'] is not None:
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.subheader("1. 结构检测")
            st.image(st.session_state['warped'], caption=f"自动矫正视图 ({st.session_state['rows']}x{st.session_state['cols']})", channels="BGR")
            
            with st.expander("🛠️ 查看 AI 看到的网格线 (Debug)"):
                debug = st.session_state['debug_info']
                if debug:
                    orig = cv2.imread(tmp_path)
                    for x in debug['v_lines']:
                        cv2.line(orig, (int(x), 0), (int(x), orig.shape[0]), (0, 255, 0), 2)
                    for y in debug['h_lines']:
                        cv2.line(orig, (0, int(y)), (orig.shape[1], int(y)), (0, 255, 0), 2)
                    for p in debug['dots']:
                        cv2.circle(orig, (p[0], p[1]), 3, (0, 0, 255), -1)
                    st.image(orig, channels="BGR", caption="绿线=识别出的行列，红点=识别出的点")

        with col2:
            st.subheader("2. 数据校对")
            st.info("👇 这里的修改会自动教 AI 变聪明！")
            df = pd.DataFrame(st.session_state['matrix'])
            edited = st.data_editor(df, key="editor", height=300, use_container_width=True)
            
            # 按钮区
            if st.button("🚀 确认并求解 (Teach & Solve)", type="primary", use_container_width=True):
                try:
                    # 1. 获取用户修正后的数据
                    final_mat = edited.fillna(-1).astype(int).values.tolist()
                    
                    # 2. 【核心升级】触发 AI 学习
                    # 在后台默默地学，界面上给个提示就好
                    with st.spinner("🧠 AI 正在根据您的修正进化..."):
                        count = st.session_state['ocr'].learn_from_feedback(final_mat)
                    
                    if count > 0:
                        st.toast(f"AI 学到了 {count} 个新字形，下次会更准！", icon="🎓")
                    
                    # 3. 求解
                    solve_it(final_mat, st.session_state['warped'])
                    
                except Exception as e:
                    st.error(f"运行错误: {e}")
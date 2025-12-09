# Project Context: Slitherlink Intelligent Hinting System
# User: Zeyu Xu (BUPT 4th Year)
# Role: AI Research Assistant

## 1. 项目目标 (Project Goal)
开发一个基于计算机视觉的数回（Slitherlink）智能辅助系统。
核心卖点不是“直接给答案”，而是 "Hinting System"（教学提示）。
目标是发表高水平论文，强调算法的完备性、鲁棒性和人机交互学习机制。

## 2. 当前技术栈 (Tech Stack)
- **Language**: Python 3.11+
- **IDE**: VS Code (Standard Venv)
- **GUI**: Streamlit (Web Interface)
- **CV**: OpenCV (opencv-python)
- **Data**: Numpy, Pandas

## 3. 已完成的核心模块 (Current Progress)
我们已经完成了 "Industrial Prototype"（工业级原型机）：
1.  **Solver (`puzzle_model.py`)**: 
    - 实现了基于邻接矩阵的数据结构。
    - 实现了 "MRV 启发式剪枝" + "增量检查" + "回溯搜索" 算法。
    - 实现了 "单一回路 (Single Loop)" 的全局拓扑检查。
    - 性能：6x6 复杂题目耗时 < 0.1s。
2.  **Vision (`digit_classifier.py`)**:
    - 实现了 "Adaptive Few-shot Learning OCR"（自适应少样本学习）。
    - 支持将学到的字形特征持久化存储到 `ocr_brain.pkl`。
    - 实现了基于 "Grid-Point Mapping" 的精准坐标映射，解决了透视畸变问题。
3.  **UI (`app.py`)**:
    - 全功能的 Streamlit 界面。
    - 实现了 "Human-in-the-loop" 交互：用户可以在表格中修正识别结果，AI 会自动隐式学习并更新模型。
    - 解决了 Streamlit 数据流回调卡顿问题，实现了丝滑交互。

## 4. 下一步计划 (Next Steps)
我们目前处于 "Phase 2: The Hinting System"。
接下来的核心任务是让 Solver 不仅仅输出最终解，而是能够生成教学提示。
- **任务 A**: 实现 "Hint Generation"（根据当前盘面，计算下一步逻辑突破口）。
- **任务 B**: 识别手写线条（检测用户当前做到哪一步了）。

## 5. 指令 (Instruction)
请基于以上上下文继续协助我开发。保持代码的高性能和学术规范性。
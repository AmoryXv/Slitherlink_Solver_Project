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
**Status: Phase 1 (Prototype) Completed ✅**

1.  **Solver (`puzzle_model.py`)**: 
    - [x] 数据结构：基于邻接矩阵 (Adjacency Matrix)。
    - [x] 算法核心：MRV 启发式剪枝 + 增量合法性检查 (Incremental Check)。
    - [x] 拓扑验证：单一回路检测 (Single Loop Check w/ DFS)。
    - [x] 性能：6x6 复杂题目耗时 < 0.1s (Real-time)。

2.  **Vision (`digit_classifier.py`)**:
    - [x] 算法：自适应少样本学习 (Adaptive Few-shot Learning)。
    - [x] 特性：支持 `ocr_brain.pkl` 持久化存储，越用越准。
    - [x] 映射：基于 Grid-Point 的精准局部映射，解决透视畸变。

3.  **UI (`app.py`)**:
    - [x] 交互：Streamlit 全栈界面。
    - [x] 亮点：Human-in-the-loop 机制（用户修正表格 -> AI 隐式学习 -> 自动更新模型）。
    - [x] 体验：解决了 Streamlit 回调卡顿问题，实现了丝滑的 Excel 式编辑。

## 4. 下一步计划 (Next Steps)
我们即将进入 **"Phase 2: The Hinting System"**。
目标是让 Solver 从“给出最终解”进化为“给出教学提示”。

- **Task A (Priority)**: 实现 **Hint Generation**。
    - 需要修改 Solver，使其能暂停在中间状态，并识别出“逻辑突破口”（即下一步该填哪条线，以及**为什么**）。
- **Task B**: 手写线条识别。
    - 识别用户当前已经画了哪些线，以便在现有基础上给提示。

## 5. 指令 (Instruction)
请读取以上代码和上下文。不要重写已完成的模块。
直接协助我设计 Task A (提示生成算法) 的接口。
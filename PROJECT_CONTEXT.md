# Project Context: Slitherlink Intelligent Hinting System
# User: Zeyu Xu (BUPT 4th Year)
# Role: AI Research Assistant

## 1. 项目目标 (Project Goal)
开发一个基于计算机视觉的数回（Slitherlink）智能辅助系统。
核心卖点不是"直接给答案"，而是 **"Hinting System"（教学提示）**。
目标是发表高水平论文，强调算法的完备性、鲁棒性（Robustness）和人机交互学习机制（Human-in-the-loop）。

## 2. 当前技术栈 (Tech Stack)
- **Language**: Python 3.11+
- **IDE**: VS Code (Standard Venv)
- **GUI**: Flask (Web Interface, bilingual Chinese/English)
- **CV & AI Core**: 
    - **OpenCV** (opencv-python): 用于传统的网格提取和图像预处理。
    - **PyTorch** (torch): 用于构建和运行 CNN 数字识别模型。
    - **Scikit-learn**: 用于 K-Means 自适应阈值聚类。
- **Data**: Numpy, Pandas

## 3. 已完成的核心模块 (Current Progress)
**Status: Phase 1 (Advanced Prototype) Completed ✅**

我们已经完成了从"基础原型"到"高精度智能系统"的迭代。

### 3.1 求解器模块 (Solver - `puzzle_model.py`)
- [x] **架构**: 基于邻接矩阵 (Adjacency Matrix) 的 MVC 分离设计。
- [x] **核心算法**: 
    - **MRV (Minimum Remaining Values)** 启发式剪枝：优先处理约束最强（如数字3或0）的边缘。
    - **增量检查 (Incremental Check)**：O(1) 复杂度的实时合法性验证。
- [x] **拓扑验证**: 实现了基于 DFS/BFS 的 **单一回路检测 (Single Loop Check)**，确保解的拓扑唯一性。
- [x] **推理日志**: 完整的逻辑推导日志系统（Rule-0/Rule-3/Vertex/Backtrack），支持 verbose 模式。

### 3.2 视觉与感知模块 (Vision - `vision_grid.py` & `ocr_engine.py`)
这是本项目技术含量最高的部分，已通过多次迭代达到工业级稳定性。

- [x] **网格结构提取 (Grid Extraction V7)**:
    - **算法**: **"Median Dictatorship" (中位数独裁)** 策略。通过统计行/列点数的中位数，强力过滤由数字 "0" 组成的伪列，解决了传统聚类算法对噪点敏感的问题。
    - **对齐**: 实现了基于透视变换 (Perspective Transform) 的精准网格拉直与切割。

- [x] **智能 OCR 引擎 (Smart OCR)**:
    - **模型**: 自研轻量级 CNN (`SimpleDigitNet`)，在混合合成数据集（粗/细字体 + 噪点增强）上训练，对印刷体具有极高鲁棒性。
    - **TTA (Test-Time Augmentation)**: 引入 **测试时增强** 技术。推理时通过多视角（原图、平移、缩放）投票机制，消除随机误判。
    - **全局动态校准 (Global Dynamic Calibration)**: 使用 **K-Means 聚类** 自动分析全图墨水密度分布，自适应计算"空位"与"数字"的二值化阈值，彻底解决光照变化导致的空位误判。

### 3.3 命令行界面 (CLI - `cli.py`)
- [x] **Core Deliverable**: 输入图片路径 → OCR 识别 → 求解 → 推导日志 + ASCII 棋盘输出。
- [x] **完整参数支持**: `--no-log`（跳过日志）、`--verbose`（完整回溯记录）。

### 3.4 Web 交互界面 (GUI - `web_app.py`)
- [x] **Flask Web UI**: 暗色主题 + Inter 字体 + Canvas 可视化 + 中英对照。
- [x] **三步流程**: 上传图片 → 矩阵校对（含原图预览对照）→ 求解结果渲染。
- [x] **Human-in-the-loop (在线微调)**: 
    - 实现了 **"Teach & Solve"** 闭环。当用户修正错误数字时，系统会自动提取对应图像样本，在后台对 CNN 模型进行**增量学习 (Online Fine-tuning)**。
    - 实现了"越用越准"的进化能力。

## 4. 下一步计划 (Next Steps)
我们即将进入 **"Phase 2: The Hinting System"**。
目标是让 Solver 从"给出最终解"进化为"给出教学提示"。

- **Task A (Priority): 提示生成 (Hint Generation)**
    - 修改 Solver 接口，使其能暂停在中间状态。
    - 识别"逻辑突破口"（Logical Bottleneck）：即下一步该填哪条线。
    - 生成解释性文本（Explanation）：解释**为什么**必须填这条线（基于周围数字约束或回路约束）。

- **Task B: 手写笔迹识别 (Handwriting Recognition)**
    - 识别用户当前在纸面上已经画了哪些线。
    - 将用户进度与 Solver 内部状态对齐，实现上下文感知的辅助。
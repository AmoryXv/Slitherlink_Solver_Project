"""
Slitherlink CLI 求解器
=====================
命令行入口：输入一张谜题图片，自动识别 → 求解 → 输出逐步推导日志和 ASCII 棋盘。

用法:
    python cli.py <图片路径>
    python cli.py test_puzzle.png
"""

import argparse
import os
import sys
import time

from ocr_engine import SlitherlinkOCR
from puzzle_model import SlitherlinkPuzzle


def print_clue_matrix(matrix, rows, cols):
    """以对齐的表格形式打印识别到的数字矩阵"""
    print(f"\n{'='*40}")
    print(f"  识别结果 ({rows}×{cols} 网格)")
    print(f"{'='*40}")
    # 列号标题
    header = "     " + "  ".join(f"{c:>2}" for c in range(cols))
    print(header)
    print("     " + "----" * cols)
    for r in range(rows):
        row_str = f" {r:>2} |"
        for c in range(cols):
            val = matrix[r][c]
            if val == -1:
                row_str += "  . "
            else:
                row_str += f"  {val} "
        print(row_str)
    print()


def print_deduction_log(log):
    """逐行打印推导日志"""
    print(f"\n{'='*60}")
    print("  逻辑推导日志 (Deduction Log)")
    print(f"{'='*60}")
    for i, entry in enumerate(log):
        # 阶段标题行直接输出，不加序号
        if entry.startswith("="):
            print(f"\n{entry}")
        else:
            print(entry)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Slitherlink 数回谜题 CLI 求解器",
        epilog="示例: python cli.py test_puzzle.png"
    )
    parser.add_argument(
        "image_path",
        help="谜题图片的文件路径 (支持 PNG / JPG)"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="跳过推导日志，只输出最终棋盘"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出完整回溯日志（默认仅输出摘要，可能非常长）"
    )
    args = parser.parse_args()

    # ── 1. 验证输入文件 ──
    if not os.path.isfile(args.image_path):
        print(f"[错误] 文件不存在: {args.image_path}")
        sys.exit(1)

    # ── 2. OCR 识别阶段 ──
    print("[1/3] 正在加载模型并识别图片...")
    try:
        ocr = SlitherlinkOCR()
        warped, matrix, rows, cols, debug_info = ocr.recognize_board_auto(args.image_path)
    except Exception as e:
        print(f"[错误] 图片识别失败: {e}")
        sys.exit(1)

    # 将 numpy 矩阵转为 python list
    clue_list = matrix.tolist()
    print_clue_matrix(clue_list, rows, cols)

    # ── 3. 求解阶段 ──
    print("[2/3] 正在求解...")
    solver = SlitherlinkPuzzle(rows, cols, clue_list)

    # 启用推理日志
    enable_log = not args.no_log
    solver._logging_enabled = enable_log
    solver._verbose_logging = args.verbose

    t_start = time.time()
    solver.apply_basic_rules()
    solved = solver.solve_backtracking()
    t_elapsed = time.time() - t_start

    # 在日志中追加回溯摘要统计
    if enable_log and solved:
        solver._log_backtrack_summary()

    # ── 4. 输出结果 ──
    if solved:
        print(f"[3/3] ✅ 求解成功！(耗时 {t_elapsed:.3f} 秒)")

        # 输出推导日志
        if enable_log and solver.deduction_log:
            print_deduction_log(solver.deduction_log)

        # 输出 ASCII 棋盘
        solver.print_board_fancy()
    else:
        print(f"[3/3] ❌ 无解 (耗时 {t_elapsed:.3f} 秒)")
        print("  可能原因：数字识别有误，请检查上方的识别结果矩阵。")

        # 即使无解也输出日志，帮助调试
        if enable_log and solver.deduction_log:
            print_deduction_log(solver.deduction_log)


if __name__ == "__main__":
    main()

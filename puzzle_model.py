import numpy as np
import sys
import time
from typing import List, Tuple, Optional

# 增加递归深度，防止复杂题目爆栈
sys.setrecursionlimit(10000)

class SlitherlinkPuzzle:
    def __init__(self, height: int, width: int, clues: List[List[int]]):
        self.height = height
        self.width = width
        self.clues = clues
        
        # 0 = 未知, 1 = 有线 (Line), 2 = 打叉 (Cross)
        self.h_edges = np.zeros((height + 1, width), dtype=int)
        self.v_edges = np.zeros((height, width + 1), dtype=int)

    def print_board_fancy(self):
        print(f"\n--- Slitherlink Board ({self.height}x{self.width}) ---")
        for r in range(self.height):
            line_str = "+"
            for c in range(self.width):
                s = self.h_edges[r][c]
                line_str += ("---" if s==1 else " x " if s==2 else "   ") + "+"
            print(line_str)
            
            row_str = ""
            for c in range(self.width):
                v = self.v_edges[r][c]
                val = self.clues[r][c]
                v_char = "|" if v==1 else "x" if v==2 else " "
                c_char = str(val) if val != -1 else "."
                row_str += f"{v_char} {c_char} "
            v_last = self.v_edges[r][self.width]
            row_str += "|" if v_last==1 else "x" if v_last==2 else " "
            print(row_str)
            
        line_str = "+"
        for c in range(self.width):
            s = self.h_edges[self.height][c]
            line_str += ("---" if s==1 else " x " if s==2 else "   ") + "+"
        print(line_str)

    # ==========================
    #   Level 1: O(1) 极速局部检查
    # ==========================
    
    def check_cell(self, r: int, c: int) -> bool:
        """只检查指定的一个格子"""
        # 越界检查：如果在边界外，认为合法
        if not (0 <= r < self.height and 0 <= c < self.width): return True
        
        target = self.clues[r][c]
        if target == -1: return True
        
        edges = [
            self.h_edges[r][c], self.h_edges[r+1][c],
            self.v_edges[r][c], self.v_edges[r][c+1]
        ]
        lines = edges.count(1)
        unknowns = edges.count(0)
        
        # 剪枝核心：
        # 1. 线太多了 -> 错
        if lines > target: return False
        # 2. 线太少了，且没空位可填了 -> 错
        if lines + unknowns < target: return False
        return True

    def check_vertex(self, r: int, c: int) -> bool:
        """只检查指定的一个顶点"""
        # 越界检查
        if not (0 <= r <= self.height and 0 <= c <= self.width): return True
        
        edges = []
        if r > 0: edges.append(self.v_edges[r-1][c])             # Up
        if r < self.height: edges.append(self.v_edges[r][c])     # Down
        if c > 0: edges.append(self.h_edges[r][c-1])             # Left
        if c < self.width: edges.append(self.h_edges[r][c])      # Right
        
        lines = edges.count(1)
        unknowns = edges.count(0)
        crosses = edges.count(2)
        
        # 拓扑规则：
        # 1. 任何顶点的线段必须是 0 或 2 条 (不能分叉，不能断头)
        if lines > 2: return False
        
        # 2. 如果已经有 1 条线，且没空位了 -> 断头路 -> 错
        if lines == 1 and unknowns == 0: return False
        
        # 3. 如果已经封死 3 面，只剩 1 面是线 -> 断头路 -> 错
        if lines == 1 and crosses == 3: return False
        
        # 4. 如果封死 3 面，剩下一面还没填 -> 那剩下一面必须也是叉 (孤立点)
        #    因为如果填线就会造成断头。这是一种前瞻剪枝。
        if crosses == 3 and unknowns == 1 and lines == 0:
            # 这里我们不修改，只做检查。修改由 apply_basic_rules 做
            pass 

        return True

    def is_valid_incremental(self, t, r, c) -> bool:
        """增量检查：改了一条边，只查它周围的邻居"""
        # 任何一条边，左右(或上下)各有一个格子，两头各有一个顶点
        if t == 'h':
            # 横边 (r, c) 影响: 格子(r-1, c), 格子(r, c)
            #               顶点(r, c), 顶点(r, c+1)
            if not self.check_cell(r-1, c): return False
            if not self.check_cell(r, c): return False
            if not self.check_vertex(r, c): return False
            if not self.check_vertex(r, c+1): return False
        else:
            # 竖边 (r, c) 影响: 格子(r, c-1), 格子(r, c)
            #               顶点(r, c), 顶点(r+1, c)
            if not self.check_cell(r, c-1): return False
            if not self.check_cell(r, c): return False
            if not self.check_vertex(r, c): return False
            if not self.check_vertex(r+1, c): return False
        return True

    # ==========================
    #   Level 2: 智能预处理 (加速器)
    # ==========================
    
    def apply_basic_rules(self):
        """
        在开始费劲的回溯前，先用简单规则把 0 和 3 填掉。
        这能极大减少搜索空间！
        """
        print("[Solver] 正在进行规则预处理...")
        changed = True
        loop_limit = 0
        while changed and loop_limit < 100: # 防止死循环
            changed = False
            loop_limit += 1
            
            # Rule 0: 0 四周全是叉
            for r in range(self.height):
                for c in range(self.width):
                    if self.clues[r][c] == 0:
                        coords = [('h',r,c), ('h',r+1,c), ('v',r,c), ('v',r,c+1)]
                        for t, tr, tc in coords:
                            if self.get_edge(t, tr, tc) == 0:
                                self.set_edge(t, tr, tc, 2)
                                changed = True
            
            # Rule 3: 3 旁边如果有叉，剩下全填线
            for r in range(self.height):
                for c in range(self.width):
                    if self.clues[r][c] == 3:
                        coords = [('h',r,c), ('h',r+1,c), ('v',r,c), ('v',r,c+1)]
                        crosses = 0
                        unknowns = []
                        for t, tr, tc in coords:
                            val = self.get_edge(t, tr, tc)
                            if val == 2: crosses += 1
                            elif val == 0: unknowns.append((t, tr, tc))
                        
                        if crosses > 0: # 只要有一个叉，剩下的全必须是线
                            for t, tr, tc in unknowns:
                                self.set_edge(t, tr, tc, 1)
                                changed = True
            
            # Vertex Rule: 顶点若 3 面不通，第 4 面必须不通
            for r in range(self.height + 1):
                for c in range(self.width + 1):
                    edges = self.get_vertex_edges(r, c) # (val, type, tr, tc)
                    cross_cnt = sum(1 for e in edges if e[0] == 2)
                    unknown_edges = [e for e in edges if e[0] == 0]
                    
                    if cross_cnt == 3 and len(unknown_edges) == 1:
                        # 封死最后一条路
                        _, t, tr, tc = unknown_edges[0]
                        self.set_edge(t, tr, tc, 2)
                        changed = True

    def get_edge(self, t, r, c):
        if t=='h': return self.h_edges[r][c]
        else: return self.v_edges[r][c]
        
    def get_vertex_edges(self, r, c):
        # 辅助函数，获取顶点连接的边信息
        res = []
        if r > 0: res.append((self.v_edges[r-1][c], 'v', r-1, c))
        if r < self.height: res.append((self.v_edges[r][c], 'v', r, c))
        if c > 0: res.append((self.h_edges[r][c-1], 'h', r, c-1))
        if c < self.width: res.append((self.h_edges[r][c], 'h', r, c))
        return res

    # ==========================
    #   Level 3: 高性能回溯 (核心)
    # ==========================

    def find_most_constrained_edge(self) -> Optional[Tuple[str, int, int]]:
        """
        [修复版] 真正的 MRV 启发式算法
        策略：评分制。优先填数字 '3' 和 '0' 旁边的边，最后才填空白区的边。
        这能避免算法在无关紧要的地方浪费时间穷举。
        """
        best_edge = None
        max_score = -float('inf')
        
        # 辅助函数：计算一条边的“紧迫程度”
        def calculate_score(t, r, c):
            score = 0
            # 1. 优先填数字旁边的
            # 检查这条边相邻的两个格子
            neighbors = []
            if t == 'h':
                if r > 0: neighbors.append((r-1, c))
                if r < self.height: neighbors.append((r, c))
            else:
                if c > 0: neighbors.append((r, c-1))
                if c < self.width: neighbors.append((r, c))
            
            has_clue = False
            for nr, nc in neighbors:
                clue = self.clues[nr][nc]
                if clue != -1:
                    has_clue = True
                    # 3 和 0 约束最强，优先处理
                    if clue == 3 or clue == 0: score += 100
                    elif clue == 1 or clue == 2: score += 50
            
            # 2. 如果旁边没数字，分就很低，最后才轮到它
            if not has_clue:
                score -= 10
                
            # 3. 拓扑启发：如果连接的顶点已经有线了，优先处理（为了连通性）
            # (这里为了速度简化，暂不计算顶点度数细节，仅靠数字启发通常足够)
            
            return score

        # 扫描所有横边
        for r in range(self.height + 1):
            for c in range(self.width):
                if self.h_edges[r][c] == 0:
                    score = calculate_score('h', r, c)
                    if score > max_score:
                        max_score = score
                        best_edge = ('h', r, c)
                        # 如果找到满分边(比如3旁边的)，直接返回，不用找了（贪婪策略）
                        if score >= 100: return best_edge

        # 扫描所有竖边
        for r in range(self.height):
            for c in range(self.width + 1):
                if self.v_edges[r][c] == 0:
                    score = calculate_score('v', r, c)
                    if score > max_score:
                        max_score = score
                        best_edge = ('v', r, c)
                        if score >= 100: return best_edge

        return best_edge

    def set_edge(self, type_str, r, c, val):
        if type_str == "h": self.h_edges[r][c] = val
        else: self.v_edges[r][c] = val

    def solve_backtracking(self) -> bool:
        # 1. 启发式选边
        target = self.find_most_constrained_edge()
        
        # Base Case: 填满了
        if target is None:
            return self.check_single_loop() # 只有全填完了才做昂贵的全局检查
        
        t, r, c = target
        
        # 2. 递归尝试
        # 尝试顺序：对于 3 旁边的边，优先试 1 (线)；对于 0 旁边的，优先试 2 (叉)
        # 这里简化为先试 线(1) 再试 叉(2)
        
        # Try Line (1)
        self.set_edge(t, r, c, 1)
        if self.is_valid_incremental(t, r, c): # 只查局部
            if self.solve_backtracking(): return True
        
        # Try Cross (2)
        self.set_edge(t, r, c, 2)
        if self.is_valid_incremental(t, r, c):
            if self.solve_backtracking(): return True
            
        # Backtrack (Reset)
        self.set_edge(t, r, c, 0)
        return False

    def check_single_loop(self) -> bool:
        """全局检查：确保只有 1 个圈"""
        total_lines = np.sum(self.h_edges == 1) + np.sum(self.v_edges == 1)
        if total_lines == 0: return False
        
        # 找起点
        start_node = None
        for r in range(self.height + 1):
            for c in range(self.width + 1):
                # 只要这个顶点连着线，就可以做起点
                edges = self.get_vertex_edges(r, c)
                if any(e[0] == 1 for e in edges):
                    start_node = (r, c)
                    break
            if start_node: break
            
        if not start_node: return False
        
        # BFS/DFS 遍历
        visited_edges = set()
        stack = [start_node]
        
        while stack:
            curr_r, curr_c = stack.pop()
            edges = self.get_vertex_edges(curr_r, curr_c)
            
            for val, t, tr, tc in edges:
                if val == 1: # 是线
                    edge_id = (t, tr, tc)
                    if edge_id not in visited_edges:
                        visited_edges.add(edge_id)
                        # 找到这条线的另一端
                        if t == 'h':
                            # 横边 (tr, tc) 连接 (tr, tc) 和 (tr, tc+1)
                            next_node = (tr, tc+1) if (tr, tc) == (curr_r, curr_c) else (tr, tc)
                        else:
                            # 竖边 (tr, tc) 连接 (tr, tc) 和 (tr+1, tc)
                            next_node = (tr+1, tc) if (tr, tc) == (curr_r, curr_c) else (tr, tc)
                        stack.append(next_node)
                        
        return len(visited_edges) == total_lines
"""
校园导航系统 — Dijkstra + A* 最短路径规划（扩展版）
支持：多模式路径（步行/自行车）、途经点规划、A* 算法与性能对比。
"""

import heapq
import math
import os
import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# 1. 校园地图数据结构
# ============================================================
class CampusMap:
    """校园地图：邻接表存储无向图，每条边含步行/自行车双权重。"""

    def __init__(self):
        self.nodes: list[str] = []
        self.name_to_idx: dict[str, int] = {}
        self.coords: list[tuple[int, int]] = []
        # adj[u] = [(v, walk_weight, bike_weight), ...]
        self.adj: list[list[tuple[int, int, int]]] = []

    def load(self, filepath: str) -> str | None:
        """从 map.txt 解析地图。边格式：地点A 地点B 步行权重 [自行车权重]。"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return f"地图文件不存在: {filepath}"
        except Exception as e:
            return f"读取文件失败: {e}"

        if not lines:
            return "地图文件为空"

        idx = 0

        # N
        try:
            n = int(lines[idx]); idx += 1
        except ValueError:
            return "第一行必须为地点数量 N（整数）"
        if n <= 0:
            return "地点数量必须大于 0"
        if idx + n > len(lines):
            return f"地点行数不足：期望 {n} 行"

        # N 行地点（含可选坐标 x y）
        self.nodes = []
        self.coords = []
        self.name_to_idx = {}
        for _ in range(n):
            parts = lines[idx].split(); idx += 1
            name = parts[0]
            if name in self.name_to_idx:
                return f"地点名称重复: {name}"
            self.name_to_idx[name] = len(self.nodes)
            self.nodes.append(name)
            if len(parts) >= 3:
                try:
                    self.coords.append((int(parts[1]), int(parts[2])))
                except ValueError:
                    return f"坐标格式错误: {lines[idx - 1]}"
            else:
                self.coords.append((0, 0))

        # M
        if idx >= len(lines):
            return "缺少路径数量 M"
        try:
            m = int(lines[idx]); idx += 1
        except ValueError:
            return "路径数量 M 必须为整数"
        if m < 0:
            return "路径数量 M 不能为负数"
        if idx + m > len(lines):
            return f"路径行数不足：期望 {m} 行"

        self.adj = [[] for _ in range(n)]

        for _ in range(m):
            parts = lines[idx].split(); idx += 1
            if len(parts) < 3:
                return f"路径格式错误: {lines[idx - 1]}"
            a, b = parts[0], parts[1]
            try:
                w_walk = int(parts[2])
            except ValueError:
                return f"步行权重必须为整数: {lines[idx - 1]}"
            if w_walk <= 0:
                return f"步行权重必须为正数: {lines[idx - 1]}"
            w_bike = w_walk  # 默认自行车=步行
            if len(parts) >= 4:
                try:
                    w_bike = int(parts[3])
                except ValueError:
                    return f"自行车权重必须为整数: {lines[idx - 1]}"
                if w_bike <= 0:
                    return f"自行车权重必须为正数: {lines[idx - 1]}"
            if a not in self.name_to_idx or b not in self.name_to_idx:
                return f"未知地点: {lines[idx - 1]}"
            u, v = self.name_to_idx[a], self.name_to_idx[b]
            self.adj[u].append((v, w_walk, w_bike))
            self.adj[v].append((u, w_walk, w_bike))

        return None

    @property
    def node_count(self) -> int:
        return len(self.nodes)


# ============================================================
# 2. 最短路径算法（Dijkstra + A*）
# ============================================================

def _weight_of(edge: tuple, mode: int) -> int:
    """从邻接表边元组 (v, walk_w, bike_w) 中取对应模式权重。mode: 0=步行, 1=自行车。"""
    return edge[mode + 1]


def dijkstra(campus: CampusMap, start: str, end: str, mode: int = 0
             ) -> tuple[list[str] | None, int, int, str | None]:
    """
    Dijkstra 最短路径。
    mode: 0=步行, 1=自行车。
    返回 (路径名称列表, 总距离, 探索节点数, 错误描述)。
    """
    if start == end:
        return None, 0, 0, "起点和终点不能相同"
    if start not in campus.name_to_idx:
        return None, 0, 0, f"起点 '{start}' 在地图中未找到"
    if end not in campus.name_to_idx:
        return None, 0, 0, f"终点 '{end}' 在地图中未找到"

    n = campus.node_count
    s = campus.name_to_idx[start]
    t = campus.name_to_idx[end]

    INF = 10 ** 18
    dist = [INF] * n
    prev = [-1] * n
    dist[s] = 0

    pq = [(0, s)]
    explored = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        explored += 1
        if u == t:
            break
        for edge in campus.adj[u]:
            v = edge[0]
            w = _weight_of(edge, mode)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[t] == INF:
        return None, 0, explored, f"从 '{start}' 到 '{end}' 没有可达路径"

    path_idx = []
    cur = t
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    path_names = [campus.nodes[i] for i in path_idx]

    return path_names, dist[t], explored, None


def a_star(campus: CampusMap, start: str, end: str, mode: int = 0
           ) -> tuple[list[str] | None, int, int, str | None]:
    """
    A* 最短路径（欧氏距离启发函数）。
    返回 (路径名称列表, 总距离, 探索节点数, 错误描述)。
    """
    if start == end:
        return None, 0, 0, "起点和终点不能相同"
    if start not in campus.name_to_idx:
        return None, 0, 0, f"起点 '{start}' 在地图中未找到"
    if end not in campus.name_to_idx:
        return None, 0, 0, f"终点 '{end}' 在地图中未找到"

    n = campus.node_count
    s = campus.name_to_idx[start]
    t = campus.name_to_idx[end]
    tx, ty = campus.coords[t]

    INF = 10 ** 18
    g = [INF] * n      # 实际距离
    prev = [-1] * n
    g[s] = 0

    # 启发函数：欧氏距离
    def h(node_idx: int) -> float:
        nx, ny = campus.coords[node_idx]
        return math.hypot(nx - tx, ny - ty)

    pq = [(h(s), 0, s)]  # (f = g + h, g, node)
    explored = 0

    while pq:
        f, d, u = heapq.heappop(pq)
        if d != g[u]:
            continue
        explored += 1
        if u == t:
            break
        for edge in campus.adj[u]:
            v = edge[0]
            w = _weight_of(edge, mode)
            nd = d + w
            if nd < g[v]:
                g[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd + h(v), nd, v))

    if g[t] == INF:
        return None, 0, explored, f"从 '{start}' 到 '{end}' 没有可达路径"

    path_idx = []
    cur = t
    while cur != -1:
        path_idx.append(cur)
        cur = prev[cur]
    path_idx.reverse()
    path_names = [campus.nodes[i] for i in path_idx]

    return path_names, g[t], explored, None


def route_with_waypoint(campus: CampusMap, start: str, waypoint: str, end: str,
                        mode: int = 0, algo: str = "dijkstra"
                        ) -> tuple[list[str] | None, int, str | None]:
    """
    途经点分段路径拼接。
    返回 (完整路径名称列表, 总距离, 错误描述)。
    """
    alg = dijkstra if algo == "dijkstra" else a_star
    seg1_path, seg1_dist, _, err = alg(campus, start, waypoint, mode)
    if err:
        return None, 0, f"第1段 ({start}→{waypoint}): {err}"
    seg2_path, seg2_dist, _, err = alg(campus, waypoint, end, mode)
    if err:
        return None, 0, f"第2段 ({waypoint}→{end}): {err}"

    full_path = seg1_path + seg2_path[1:]  # 去重 waypoint
    return full_path, seg1_dist + seg2_dist, None


# ============================================================
# 3. Tkinter 图形界面
# ============================================================
class CampusNavigationApp:
    """校园导航系统主窗口。"""

    NODE_RADIUS = 28
    NODE_COLOR = "#4A90D9"
    NODE_HIGHLIGHT = "#E74C3C"
    EDGE_COLOR = "#95A5A6"
    EDGE_HIGHLIGHT = "#E74C3C"
    EDGE_WIDTH = 2
    EDGE_HIGHLIGHT_WIDTH = 4

    MODE_LABELS = ["步行", "自行车"]

    def __init__(self, root: tk.Tk, map_file: str):
        self.root = root
        self.root.title("校园导航系统")
        self.root.geometry("960x620")
        self.root.resizable(False, False)

        self.campus = CampusMap()
        self.current_path: list[str] | None = None
        self.current_mode = 0  # 0=步行, 1=自行车

        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧 — 地图画布
        self.canvas = tk.Canvas(
            self.main_frame, width=640, height=580,
            bg="#F5F6FA", highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))

        # 右侧 — 控制面板（加宽以适应新控件）
        self.panel = ttk.Frame(self.main_frame, width=280)
        self.panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.panel.pack_propagate(False)

        self._build_panel()
        self._load_map(map_file)

    # ---------- 控制面板 ----------
    def _build_panel(self):
        ttk.Label(self.panel, text="校园导航系统",
                  font=("Microsoft YaHei", 15, "bold")).pack(pady=(0, 12))

        # 交通模式
        ttk.Label(self.panel, text="交通模式:", font=("Microsoft YaHei", 10)).pack(anchor=tk.W)
        mode_frame = ttk.Frame(self.panel)
        mode_frame.pack(fill=tk.X, pady=(2, 10))
        self.mode_var = tk.IntVar(value=0)
        for i, label in enumerate(self.MODE_LABELS):
            ttk.Radiobutton(
                mode_frame, text=label, variable=self.mode_var, value=i,
                command=self._on_mode_changed
            ).pack(side=tk.LEFT, padx=(0, 12))

        # 起点
        ttk.Label(self.panel, text="起点:", font=("Microsoft YaHei", 10)).pack(anchor=tk.W)
        self.start_var = tk.StringVar()
        self.start_combo = ttk.Combobox(
            self.panel, textvariable=self.start_var,
            state="readonly", font=("Microsoft YaHei", 10)
        )
        self.start_combo.pack(fill=tk.X, pady=(2, 8))

        # 途经点
        ttk.Label(self.panel, text="途经点 (可选):", font=("Microsoft YaHei", 10)).pack(anchor=tk.W)
        self.waypoint_var = tk.StringVar(value="无")
        self.waypoint_combo = ttk.Combobox(
            self.panel, textvariable=self.waypoint_var,
            state="readonly", font=("Microsoft YaHei", 10)
        )
        self.waypoint_combo.pack(fill=tk.X, pady=(2, 8))

        # 终点
        ttk.Label(self.panel, text="终点:", font=("Microsoft YaHei", 10)).pack(anchor=tk.W)
        self.end_var = tk.StringVar()
        self.end_combo = ttk.Combobox(
            self.panel, textvariable=self.end_var,
            state="readonly", font=("Microsoft YaHei", 10)
        )
        self.end_combo.pack(fill=tk.X, pady=(2, 10))

        # 查询按钮
        self.search_btn = ttk.Button(
            self.panel, text="查询最短路径", command=self._on_search
        )
        self.search_btn.pack(fill=tk.X, pady=(0, 6))

        # 查看所有地点
        self.list_btn = ttk.Button(
            self.panel, text="查看所有地点", command=self._on_list_places
        )
        self.list_btn.pack(fill=tk.X, pady=(0, 12))

        # 结果展示
        ttk.Label(self.panel, text="查询结果:", font=("Microsoft YaHei", 10, "bold")).pack(anchor=tk.W)
        self.result_frame = ttk.Frame(self.panel)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(
            self.result_frame, height=14, wrap=tk.WORD,
            font=("Microsoft YaHei", 10), state=tk.DISABLED,
            bg="#FFFFFF", relief=tk.SOLID, borderwidth=1
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # 图例
        ttk.Label(
            self.panel,
            text="● 普通节点  ● 路径高亮  — 最短路径",
            font=("Microsoft YaHei", 8), foreground="#7F8C8D"
        ).pack(pady=(8, 0))

    # ---------- 地图加载 ----------
    def _load_map(self, map_file: str):
        err = self.campus.load(map_file)
        if err:
            messagebox.showerror("地图加载失败", err)
            self.root.destroy()
            return

        if all(x == 0 and y == 0 for x, y in self.campus.coords):
            self._auto_layout()

        names = self.campus.nodes
        waypoint_names = ["无"] + names
        self.start_combo["values"] = names
        self.end_combo["values"] = names
        self.waypoint_combo["values"] = waypoint_names
        if names:
            self.start_combo.current(0)
            self.end_combo.current(len(names) - 1 if len(names) > 1 else 0)
            self.waypoint_combo.current(0)

        self._draw_map()

    def _auto_layout(self):
        n = self.campus.node_count
        cx, cy = 320, 290
        r = 200
        self.campus.coords = []
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            self.campus.coords.append((x, y))

    # ---------- 地图绘制 ----------
    def _draw_map(self):
        self.canvas.delete("all")

        # 边（去重）
        drawn_edges = set()
        for u in range(self.campus.node_count):
            for edge in self.campus.adj[u]:
                v = edge[0]
                pair = (min(u, v), max(u, v))
                if pair in drawn_edges:
                    continue
                drawn_edges.add(pair)
                w_walk = edge[1]
                w_bike = edge[2]
                self._draw_edge(u, v, w_walk, w_bike, highlight=False)

        # 节点
        for i, name in enumerate(self.campus.nodes):
            self._draw_node(i, name, highlight=False)

        if self.current_path:
            self._highlight_path()

    def _draw_node(self, idx: int, name: str, highlight: bool):
        x, y = self.campus.coords[idx]
        r = self.NODE_RADIUS
        color = self.NODE_HIGHLIGHT if highlight else self.NODE_COLOR
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill=color, outline="", tags="node")
        self.canvas.create_text(x, y, text=name, fill="white",
                                font=("Microsoft YaHei", 9, "bold"), tags="node")

    def _draw_edge(self, u: int, v: int, w_walk: int, w_bike: int, highlight: bool):
        x1, y1 = self.campus.coords[u]
        x2, y2 = self.campus.coords[v]
        color = self.EDGE_HIGHLIGHT if highlight else self.EDGE_COLOR
        width = self.EDGE_HIGHLIGHT_WIDTH if highlight else self.EDGE_WIDTH
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, tags="edge")

        # 显示当前模式对应的权重，多模式时标注单位
        if w_walk == w_bike:
            label = str(w_walk)
        else:
            active_w = w_walk if self.current_mode == 0 else w_bike
            unit = "分" if self.current_mode == 0 else "分"
            label = f"{active_w}{unit}"
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2 - 8
        self.canvas.create_text(
            mx, my, text=label,
            fill="#2C3E50", font=("Microsoft YaHei", 8),
            tags="edge"
        )

    def _highlight_path(self):
        if not self.current_path or len(self.current_path) < 2:
            return
        path_idx = [self.campus.name_to_idx[n] for n in self.current_path]
        drawn = set()
        for i in range(len(path_idx) - 1):
            u, v = path_idx[i], path_idx[i + 1]
            pair = (min(u, v), max(u, v))
            if pair not in drawn:
                drawn.add(pair)
                for edge in self.campus.adj[u]:
                    if edge[0] == v:
                        self._draw_edge(u, v, edge[1], edge[2], highlight=True)
                        break
        for idx in path_idx:
            self._draw_node(idx, self.campus.nodes[idx], highlight=True)

    # ---------- 模式切换 ----------
    def _on_mode_changed(self):
        self.current_mode = self.mode_var.get()
        self.current_path = None
        self._draw_map()
        self._set_result("")

    # ---------- 查询 ----------
    def _on_search(self):
        start = self.start_var.get().strip()
        end = self.end_var.get().strip()
        waypoint = self.waypoint_var.get().strip()
        mode = self.current_mode
        mode_label = self.MODE_LABELS[mode]

        if not start or not end:
            self._set_result("请选择起点和终点")
            return

        use_waypoint = waypoint and waypoint != "无"

        # 途经点模式：分段拼接
        if use_waypoint:
            if waypoint == start or waypoint == end:
                self._set_result("途经点不能与起点或终点相同", is_error=True)
                self.current_path = None
                self._draw_map()
                return
            full_path, total_dist, err = route_with_waypoint(
                self.campus, start, waypoint, end, mode, "dijkstra"
            )
            self.current_path = full_path
            self._draw_map()
            if err:
                self._set_result(err, is_error=True)
                return
            route_str = " → ".join(full_path)
            self._set_result(
                f"【途经点模式 — Dijkstra】\n"
                f"最短路径:\n{route_str}\n\n"
                f"总距离: {total_dist} ({mode_label})\n"
                f"途经: {waypoint}"
            )
            return

        # 无途经点：同时运行 Dijkstra 和 A* 对比
        if start == end:
            self._set_result("起点和终点不能相同", is_error=True)
            self.current_path = None
            self._draw_map()
            return

        d_path, d_dist, d_explored, d_err = dijkstra(self.campus, start, end, mode)
        a_path, a_dist, a_explored, a_err = a_star(self.campus, start, end, mode)

        if d_err:
            self.current_path = None
            self._draw_map()
            self._set_result(d_err, is_error=True)
            return

        # 以 Dijkstra 结果高亮
        self.current_path = d_path
        self._draw_map()

        route_str = " → ".join(d_path)
        self._set_result(
            f"【{mode_label}模式 — 算法对比】\n"
            f"最短路径:\n{route_str}\n\n"
            f"总距离: {d_dist}\n\n"
            f"━━ 性能对比 ━━\n"
            f"  Dijkstra 探索节点: {d_explored}\n"
            f"  A*       探索节点: {a_explored}\n"
            f"  (结果一致: {'是' if d_dist == a_dist else '否'})"
        )

    def _on_list_places(self):
        """列出地图中所有地点。"""
        count = self.campus.node_count
        places = "\n".join(f"  • {name}" for name in self.campus.nodes)
        self._set_result(f"地图中共有 {count} 个地点:\n{places}")

    def _set_result(self, msg: str, is_error: bool = False):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", msg)
        if is_error:
            self.result_text.tag_add("error", "1.0", "end")
            self.result_text.tag_config("error", foreground="#E74C3C")
        self.result_text.config(state=tk.DISABLED)


# ============================================================
# 4. 程序入口
# ============================================================
def main():
    map_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.txt")
    root = tk.Tk()
    CampusNavigationApp(root, map_file)
    root.mainloop()


if __name__ == "__main__":
    main()

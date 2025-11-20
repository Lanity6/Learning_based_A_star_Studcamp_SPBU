import math
import heapq
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

@dataclass
class AstarOutput:
    histories: torch.Tensor
    paths: torch.Tensor
    intermediate_results: Optional[List[dict]] = None
    g: Optional[torch.Tensor] = None

def diag_heuristic(a: int, b: int, size: int) -> float:
    ax, ay = divmod(a, size)
    bx, by = divmod(b, size)
    dx, dy = abs(ax - bx), abs(ay - by)
    return min(dx, dy) * math.sqrt(2) + abs(dx - dy)

def los(x0, y0, x1, y1, occ):
    """
    Line-of-sight check using integer grid (Bresenham-like).
    occ: HxW grid map where 1 = obstacle, 0 = free
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    if dx >= dy:
        err = dx / 2
        while x0 != x1:
            err -= dy
            if err < 0:
                y0 += sy
                err += dx
            x0 += sx
            if occ[x0, y0] == 0:
                return False
    else:
        err = dy / 2
        while y0 != y1:
            err -= dx
            if err < 0:
                x0 += sx
                err += dy
            y0 += sy
            if occ[x0, y0] == 0:
                return False
    return True

def neighbors(idx: int, size: int):
    x, y = divmod(idx, size)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < size and 0 <= ny < size:
            yield nx * size + ny, math.sqrt(2) if dx != 0 and dy != 0 else 1.0

class ThetaStarPlanner(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cost_maps, start_maps, goal_maps, obstacles_maps):
        device = cost_maps.device
        B, _, H, W = start_maps.shape
        size = H

        paths = torch.zeros(B, 1, H, W, device=device)
        histories = torch.zeros(B, 1, H, W, device=device)
        g_all = torch.full((B, 1, H, W), float('inf'), device=device)

        for b in range(B):
            occ = obstacles_maps[b, 0] < 0.5  # 0 = free, 1 = wall
            start_idx = torch.argmax(start_maps[b, 0]).item()
            goal_idx = torch.argmax(goal_maps[b, 0]).item()

            # A* initialization
            g = {i: float('inf') for i in range(size * size)}
            parent = {i: None for i in range(size * size)}

            g[start_idx] = 0
            parent[start_idx] = start_idx

            open_set = []
            heapq.heappush(open_set, (0 + diag_heuristic(start_idx, goal_idx, size), start_idx))

            closed = set()

            while open_set:
                _, u = heapq.heappop(open_set)
                if u in closed:
                    continue
                closed.add(u)

                ux, uy = divmod(u, size)
                histories[b, 0, ux, uy] = 1

                if u == goal_idx:
                    break

                for v, cost_uv in neighbors(u, size):
                    vx, vy = divmod(v, size)
                    if not occ[vx, vy]:
                        continue

                    # Theta*: los
                    p = parent[u]
                    if p is not None:
                        px, py = divmod(p, size)
                        if los(px, py, vx, vy, occ):
                            # прямое соединение
                            cost_pv = math.dist((px, py), (vx, vy))
                            if g[p] + cost_pv < g[v]:
                                g[v] = g[p] + cost_pv
                                parent[v] = p
                                f = g[v] + diag_heuristic(v, goal_idx, size)
                                heapq.heappush(open_set, (f, v))
                                continue

                    if g[u] + cost_uv < g[v]:
                        g[v] = g[u] + cost_uv
                        parent[v] = u
                        f = g[v] + diag_heuristic(v, goal_idx, size)
                        heapq.heappush(open_set, (f, v))

            # Backtracking
            path = []
            cur = goal_idx
            if parent[cur] is not None:
                while True:
                    path.append(cur)
                    if cur == parent[cur]:
                        break
                    cur = parent[cur]
                    if cur is None:
                        break

            for idx in path:
                x, y = divmod(idx, size)
                paths[b, 0, x, y] = 1

            for i in range(size * size):
                x, y = divmod(i, size)
                g_all[b, 0, x, y] = g[i]

        return AstarOutput(histories=histories, paths=paths, g=g_all)
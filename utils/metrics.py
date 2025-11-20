from dataclasses import dataclass
from typing import Optional, List

import torch
import numpy as np

EPS = 1e-10


@dataclass
class Metrics:
    p_opt: float
    p_exp: float
    h_mean: float
    pcost_dif: float
    pcost_dif_list: list = None
    p_exp_list: list = None
    abs_cost: list = None
    abs_exp: list = None
    time_ratio: float = None           # <--- NEW
    time_ratio_list: list = None       # <--- NEW

    def __repr__(self):
        s  = f"optimality: {self.p_opt:0.3f}, "
        s += f"efficiency: {self.p_exp:0.3f}, "
        s += f"h_mean: {self.h_mean:0.3f}, "
        s += f"cost_diff: {self.pcost_dif:0.3f}, "
        if self.time_ratio is not None:
            s += f"time_ratio: {self.time_ratio:0.3f}"
        return s


@dataclass
class AstarOutput:
    histories: torch.tensor
    paths: torch.tensor
    intermediate_results: Optional[List[dict]] = None
    g: Optional[torch.tensor] = None


def calc_metrics(na_outputs: AstarOutput,
                 va_outputs: AstarOutput,
                 time_na: np.ndarray,
                 time_va: np.ndarray) -> Metrics:
    """
    Calculate metrics for problem instances each with a single starting point.

    Args:
        na_outputs: outputs from Neural A*
        va_outputs: outputs from vanilla A*
        time_na: array of runtimes for Neural A*
        time_va: array of runtimes for vanilla A*
    """

    # --- optimality ---
    pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    pathlen_na = na_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    p_opt = (pathlen_astar == pathlen_na).mean()

    # --- path cost ---
    pathcost_astar = torch.amax(va_outputs.paths * va_outputs.g, dim=(1, 2, 3)).detach().cpu().numpy()
    pathcost_na = torch.amax(na_outputs.paths * na_outputs.g, dim=(1, 2, 3)).detach().cpu().numpy()
    pcost_dif_list = (pathcost_na / (pathcost_astar + EPS))
    pcost_dif = pcost_dif_list.mean()

    # --- expansions ---
    exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    exp_na = na_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    p_exp_list = (exp_astar - exp_na) / (exp_astar + EPS)
    p_exp = p_exp_list.mean()

    # --- harmonic mean ---
    h_mean = 2.0 / (1.0 / (p_opt + EPS) + 1.0 / (p_exp + EPS))

    # ======================================================
    # NEW: Time comparison metric
    # ======================================================

    # time_ratio = (NA_time / A*_time)
    time_ratio_list = (time_na + EPS) / (time_va + EPS)
    time_ratio = time_ratio_list.mean()

    # ======================================================

    return Metrics(
        p_opt=p_opt,
        p_exp=1 - p_exp,
        h_mean=h_mean,
        pcost_dif=pcost_dif,
        pcost_dif_list=pcost_dif_list.tolist(),
        p_exp_list=(1 - p_exp_list).tolist(),
        abs_cost=pathcost_na.tolist(),
        abs_exp=exp_na.tolist(),
        time_ratio=time_ratio,
        time_ratio_list=time_ratio_list.tolist(),
    )

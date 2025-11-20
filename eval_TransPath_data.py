import time
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.hmaps import GridData
from models.autoencoder import Autoencoder
from modules.planners import DifferentiableDiagAstar, get_diag_heuristic
from modules.Theta_star_planner import ThetaStarPlanner

def main(mode, state_dict_path, hardness_limit=1.05):
    device = 'cuda'

    # ===============================
    # Load dataset
    # ===============================
    test_data = GridData(
        path='./TransPath_data/test',
        mode=mode
    )
    test_loader = DataLoader(
        test_data, batch_size=128, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ===============================
    # Load model
    # ===============================
    model = Autoencoder(mode=mode)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()

    # ===============================
    # Planners
    # ===============================
    vanilla_planner = DifferentiableDiagAstar(mode='default', h_w=1).to(device)

    if mode == 'cf':
        learned_planner = DifferentiableDiagAstar(mode='k').to(device)
    else:
        learned_planner = DifferentiableDiagAstar(mode=mode, f_w=100).to(device)

    theta_planner = ThetaStarPlanner().to(device)

    # ===============================
    # Metrics storage
    # ===============================
    exp_ratio = []
    cost_ratio = []
    hardness = []

    # timing
    time_vanilla = []
    time_learned = []
    time_theta = []

    # ===============================
    # Main evaluation loop
    # ===============================
    for batch in tqdm(test_loader):

        map_design, start, goal, gt_heatmap = batch
        map_design = map_design.to(device)
        start = start.to(device)
        goal = goal.to(device)

        # Prepare NN input
        if mode == 'f':
            nn_input = torch.cat([map_design, start + goal], dim=1).to(device)
        else:
            nn_input = torch.cat([map_design, goal], dim=1).to(device)

        with torch.no_grad():
            # ----------------------------------------------------
            # Predict heuristic heatmap
            # ----------------------------------------------------
            predictions = (model(nn_input) + 1) / 2

            # ----------------------------------------------------
            # VANILLA A*
            # ----------------------------------------------------
            t0 = time.time()
            vanilla_out = vanilla_planner(
                (map_design == 0).float(),
                start, goal,
                (map_design == 0).float()
            )
            time_vanilla.append(time.time() - t0)

            # ----------------------------------------------------
            # LEARNED A*
            # ----------------------------------------------------
            t0 = time.time()
            learned_out = learned_planner(
                predictions,
                start, goal,
                (map_design == 0).float()
            )
            time_learned.append(time.time() - t0)

            # ----------------------------------------------------
            # THETA*
            # ----------------------------------------------------
            t0 = time.time()
            theta_out = theta_planner(
                predictions,       # Theta* не использует cost_maps → но передаём для совместимости
                start, goal,
                (map_design == 0).float()
            )
            time_theta.append(time.time() - t0)

        # ====================================================
        # Collect metrics for A* vs Learned A*
        # ====================================================
        exp_ratio.append(
            learned_out.histories.sum((-1, -2, -3)) /
            vanilla_out.histories.sum((-1, -2, -3))
        )

        learn_cost = (learned_out.g * goal).sum((-1, -2, -3))
        vanilla_cost = (vanilla_out.g * goal).sum((-1, -2, -3))
        cost_ratio.append(learn_cost / vanilla_cost)

        start_h = (get_diag_heuristic(goal[:, 0]) * start[:, 0]).sum((-1, -2))
        hardness.append(vanilla_cost / start_h)

    # ====================================================
    # Aggregate metrics
    # ====================================================
    exp_ratio = torch.cat(exp_ratio)
    cost_ratio = torch.cat(cost_ratio)
    hardness = torch.cat(hardness)

    mask = (hardness >= hardness_limit).float()
    n = mask.sum()

    exp_ratio = (exp_ratio * mask).sum() / n
    cost_ratio = (cost_ratio * mask).sum() / n

    # ---------------------------
    # TIME METRICS
    # ---------------------------
    t_vanilla = np.mean(time_vanilla)
    t_learned = np.mean(time_learned)
    t_theta = np.mean(time_theta)

    print("\n========= RESULTS =========")
    print(f"expansion ratio (learned / vanilla): {exp_ratio.item():.4f}")
    print(f"cost ratio      (learned / vanilla): {cost_ratio.item():.4f}")
    print()
    print(f"vanilla A* time: {t_vanilla:.5f} s")
    print(f"learned A* time: {t_learned:.5f} s   ({t_learned / t_vanilla:.3f}x)")
    print(f"Theta* time:     {t_theta:.5f} s   ({t_theta / t_vanilla:.3f}x)")
    print("============================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf'], default='f')
    parser.add_argument('--weights_path', type=str, default='./weights/focal.pth')
    parser.add_argument('--seed', type=int, default=39)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    main(mode=args.mode, state_dict_path=args.weights_path)

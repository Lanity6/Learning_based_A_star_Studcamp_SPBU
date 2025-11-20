from models.autoencoder import Autoencoder
from data.hmaps import GridData
from modules.planners import DifferentiableDiagAstar, get_diag_heuristic

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import time
import argparse


def main(mode, state_dict_path, hardness_limit=1.05):
    device = 'cuda'

    # -----------------------------
    # Load test dataset
    # -----------------------------
    test_data = GridData(
        path='./TransPath_data/test',
        mode=mode
    )
    test_dataloader = DataLoader(test_data, batch_size=256,
                                 shuffle=False, num_workers=0, pin_memory=True)

    # -----------------------------
    # Load model
    # -----------------------------
    model = Autoencoder(mode=mode)
    model.load_state_dict(torch.load(state_dict_path))
    model.to(device)
    model.eval()

    # -----------------------------
    # Planners (vanilla + learnable)
    # -----------------------------
    vanilla_planner = DifferentiableDiagAstar(mode='default', h_w=1)
    learnable_planner = DifferentiableDiagAstar(
        mode='k' if mode == 'cf' else mode,
        f_w=100
    )

    vanilla_planner.to(device)
    learnable_planner.to(device)

    # -----------------------------
    # Metrics
    # -----------------------------
    expansions_ratio = []
    cost_ratio = []
    hardness = []

    # --- NEW ---
    time_ratio_list = []
    time_vanilla_list = []
    time_learned_list = []

    # -----------------------------
    # Loop over batches
    # -----------------------------
    for batch in tqdm(test_dataloader):
        with torch.no_grad():

            map_design, start, goal, gt_heatmap = batch

            # input for NN
            if mode == 'f':
                inputs = torch.cat([map_design, start + goal], dim=1)
            else:
                inputs = torch.cat([map_design, goal], dim=1)

            inputs = inputs.to(device)
            map_design = map_design.to(device)
            start = start.to(device)
            goal = goal.to(device)

            # -------------------------
            # Prediction from Autoencoder
            # -------------------------
            predictions = (model(inputs) + 1) / 2

            # -------------------------
            # TIMING for LEARNED PLANNER
            # -------------------------
            t0 = time.time()
            learn_outputs = learnable_planner(
                predictions,
                start,
                goal,
                ((map_design == 0) * 1.)
            )
            t_learn = time.time() - t0

            # -------------------------
            # TIMING for VANILLA A*
            # -------------------------
            t0 = time.time()
            vanilla_outputs = vanilla_planner(
                ((map_design == 0) * 1.),
                start,
                goal,
                ((map_design == 0) * 1.)
            )
            t_vanilla = time.time() - t0

            # save timing
            time_learned_list.append(t_learn)
            time_vanilla_list.append(t_vanilla)
            time_ratio_list.append(t_learn / (t_vanilla + 1e-9))

            # -------------------------
            # EXPANSION RATIO
            # -------------------------
            expansions_ratio.append(
                (learn_outputs.histories.sum((-1, -2, -3))) /
                (vanilla_outputs.histories.sum((-1, -2, -3)))
            )

            # -------------------------
            # COST RATIO
            # -------------------------
            learn_costs = (learn_outputs.g * goal).sum((-1, -2, -3))
            vanilla_costs = (vanilla_outputs.g * goal).sum((-1, -2, -3))
            cost_ratio.append(learn_costs / vanilla_costs)

            # -------------------------
            # HARDNESS
            # -------------------------
            start_heur = (
                get_diag_heuristic(goal[:, 0]) * start[:, 0]
            ).sum((-1, -2))

            hardness.append(vanilla_costs / start_heur)

    # -----------------------------
    # Convert to tensors
    # -----------------------------
    expansions_ratio = torch.cat(expansions_ratio, dim=0)
    cost_ratio = torch.cat(cost_ratio, dim=0)
    hardness = torch.cat(hardness, dim=0)

    # -----------------------------
    # Apply hardness filter
    # -----------------------------
    mask = (hardness >= hardness_limit).float()
    n = mask.sum()

    expansions_ratio = (expansions_ratio * mask).sum() / n
    cost_ratio = (cost_ratio * mask).sum() / n

    # -----------------------------
    # TIME metrics
    # -----------------------------
    time_ratio = np.mean(time_ratio_list)
    time_vanilla_mean = np.mean(time_vanilla_list)
    time_learned_mean = np.mean(time_learned_list)

    # -----------------------------
    # Output
    # -----------------------------
    print(f"\n===== RESULTS =====")
    print(f"expansions_ratio: {expansions_ratio.item():.4f}")
    print(f"cost_ratio:       {cost_ratio.item():.4f}")
    print(f"time_ratio:       {time_ratio:.4f}  (learned / vanilla)")
    print(f"vanilla_time_avg: {time_vanilla_mean:.6f} s")
    print(f"learned_time_avg: {time_learned_mean:.6f} s")
    print("====================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf'], default='f')
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--weights_path', type=str, default='./weights/focal.pth')

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    main(
        mode=args.mode,
        state_dict_path=args.weights_path,
    )

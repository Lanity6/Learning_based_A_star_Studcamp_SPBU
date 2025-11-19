import torch

def build_loss_mask(maps, starts, goals, mode):
    mask = torch.ones_like(maps)

    mask *= (maps == 0)
    mask *= (goals == 0)
    if mode == "f":
        mask *= (starts == 0)
    mask = mask.to('cuda:0')
    return mask.float()
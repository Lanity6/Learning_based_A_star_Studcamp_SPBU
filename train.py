from models.autoencoder import Autoencoder, PathLogger
from data.hmaps import GridData

import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

import argparse
import multiprocessing


def main(mode, run_name, proj_name, batch_size, max_epochs):
    train_data = GridData(
        path='./Dataset_sanity_check/train',
        mode=mode
    )
    val_data = GridData(
        path='./Dataset_sanity_check/val',
        mode=mode
    )
    resolution = (train_data.img_size, train_data.img_size)
    train_dataloader = DataLoader(  train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=6,
                                    pin_memory=True)
    val_dataloader = DataLoader(    val_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=6,
                                    pin_memory=True)

    samples = next(iter(val_dataloader))
    
    model = Autoencoder(mode=mode, resolution=resolution)
    callback = PathLogger(samples, mode=mode)
    wandb_logger = WandbLogger(project=proj_name, name=f'{run_name}_{mode}', log_model='all')
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/without_loss_mask",
        filename="best",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="auto",
        max_epochs=max_epochs,
        deterministic=False,
        callbacks=[callback, checkpoint_callback]
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    best_ckpt = checkpoint_callback.best_model_path

    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location="cuda")
        state_dict = ckpt["state_dict"]
        if mode == "f":
            torch.save(state_dict, "weights/best_weights_without_loss_mask_f.pth")
            print("Saved best model to best_weights_without_loss_mask_f.pth")
        else:
            torch.save(state_dict, "weights/best_weights_without_loss_mask_cf.pth")
            print("Saved best model to best_weights_without_loss_mask_cf.pth")
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['f', 'cf'], default='cf')
    parser.add_argument('--run_name', type=str, default='default')
    parser.add_argument('--proj_name', type=str, default='TransPath_runs')
    parser.add_argument('--seed', type=int, default=39)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=160)
    
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    torch.set_float32_matmul_precision('high') #fix for tensor blocks warning with new video card
    main(
        mode=args.mode,
        run_name=args.run_name,
        proj_name=args.proj_name,
        batch_size=args.batch,
        max_epochs=args.epoch,
    )

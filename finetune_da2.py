import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.expanduser('~/depth_anything_V2/Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2


class MaritimeDepthDataset(Dataset):
    def __init__(self, split_dir, input_size=518):
        self.rgb_dir   = Path(split_dir) / 'rgb'
        self.depth_dir = Path(split_dir) / 'depth'
        self.input_size = input_size
        self.frames = sorted([f.stem for f in self.rgb_dir.glob('*.png')])
        print(f"Dataset: {len(self.frames)} frames from {split_dir}")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        stem = self.frames[idx]

        # Load RGB
        rgb = cv2.imread(str(self.rgb_dir / f"{stem}.png"))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size))
        rgb = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        rgb  = (rgb - mean) / std
        rgb  = torch.from_numpy(rgb.transpose(2, 0, 1)).float()

        # Load depth (16-bit PNG in millimeters)
        depth = cv2.imread(str(self.depth_dir / f"{stem}.png"),
                           cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_NEAREST)
        depth = depth.astype(np.float32) / 1000.0  # mm → meters
        depth = torch.from_numpy(depth).float()

        return rgb, depth


def silog_loss(pred, target, variance_focus=0.85):
    """Scale-invariant log loss with robust masking."""
    mask = (
        (target > 0.5) &
        (pred > 0.5) &
        torch.isfinite(pred) &
        torch.isfinite(target)
    )
    if mask.sum() < 100:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    p = torch.clamp(pred[mask], min=0.001)
    t = torch.clamp(target[mask], min=0.001)
    d = torch.log(p) - torch.log(t)
    loss = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2) + 1e-6)
    return loss


def gradient_loss(pred, target):
    """Edge-preserving gradient loss."""
    mask = (
        (target > 0.5) &
        torch.isfinite(pred) &
        torch.isfinite(target)
    ).float()

    def gradient(x):
        dy = x[:, 1:, :] - x[:, :-1, :]
        dx = x[:, :, 1:] - x[:, :, :-1]
        return dx, dy

    pred_dx,   pred_dy   = gradient(pred)
    target_dx, target_dy = gradient(target)

    loss_x = (torch.abs(pred_dx - target_dx) * mask[:, :, 1:]).mean()
    loss_y = (torch.abs(pred_dy - target_dy) * mask[:, 1:, :]).mean()
    return loss_x + loss_y


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    checkpoint = os.path.expanduser(
        f'~/depth_anything_V2/Depth-Anything-V2/checkpoints/'
        f'depth_anything_v2_{args.encoder}.pth'
    )
    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model = model.to(device)
    print(f"Loaded pretrained: {checkpoint}")

    dataset_root = os.path.expanduser('~/depth_anything_V2/training_data/dataset')
    train_dataset = MaritimeDepthDataset(f"{dataset_root}/train", args.input_size)
    val_dataset   = MaritimeDepthDataset(f"{dataset_root}/val",   args.input_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # ── Training ──────────────────────────────────────────
        model.train()
        train_losses = []

        for rgb, depth in tqdm(train_loader,
                               desc=f"Epoch {epoch+1}/{args.epochs} [train]"):
            rgb   = rgb.to(device)
            depth = depth.to(device)

            pred = model(rgb)

            loss_si   = silog_loss(pred, depth)
            loss_grad = gradient_loss(pred, depth)
            loss = loss_si + 0.5 * loss_grad

            if not torch.isfinite(loss):
                continue

            optimizer.zero_grad()
            loss.backward()

            # Skip step if gradients are NaN
            has_nan_grad = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            )
            if has_nan_grad:
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation ────────────────────────────────────────
        model.eval()
        val_losses = []

        with torch.no_grad():
            for rgb, depth in tqdm(val_loader,
                                   desc=f"Epoch {epoch+1}/{args.epochs} [val]"):
                rgb   = rgb.to(device)
                depth = depth.to(device)
                pred  = model(rgb)
                loss  = silog_loss(pred, depth)
                if torch.isfinite(loss):
                    val_losses.append(loss.item())

        train_loss = np.mean(train_losses) if train_losses else float('nan')
        val_loss   = np.mean(val_losses)   if val_losses   else float('nan')
        scheduler.step()

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.6f} "
              f"train_batches={len(train_losses)}")

        # Save best
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir,
                                     f'da2_maritime_{args.encoder}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model: {save_path}")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output_dir,
                                     f'da2_maritime_{args.encoder}_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',    type=str,   default='vitl',
                        choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch-size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=5e-6)
    parser.add_argument('--input-size', type=int,   default=518)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.expanduser(
                            '~/depth_anything_V2/training_data/checkpoints'))
    args = parser.parse_args()
    train(args)
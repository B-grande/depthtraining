import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from tqdm import tqdm


def is_water_frame(rgb_path):
    """Returns True if frame contains significant water in lower half."""
    img = cv2.imread(str(rgb_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    lower = img[h//2:, :]
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

    # Water: blue-green range
    water_mask = cv2.inRange(hsv,
                              np.array([85, 30, 20]),
                              np.array([130, 255, 255]))

    water_ratio = water_mask.sum() / (255 * water_mask.size)
    return water_ratio > 0.15


def scale_to_metric(args):
    depth_dir  = Path(args.depth_dir)
    raw_dir    = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scale factor
    with open(args.scale_factor, 'r') as f:
        scale = float(f.read().strip())
    print(f"Using scale factor: {scale:.4f}")

    depth_maps = sorted(depth_dir.glob('*.png'))
    print(f"Found {len(depth_maps)} depth maps")

    saved = 0
    skipped_quality = 0
    skipped_water = 0

    for depth_path in tqdm(depth_maps):
        stem = depth_path.stem
        rgb_path = raw_dir / f"{stem}.png"

        # Check water content
        if not is_water_frame(rgb_path):
            skipped_water += 1
            continue

        depth_8bit = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth_8bit is None:
            skipped_quality += 1
            continue

        # Convert to metric
        depth_norm   = depth_8bit.astype(np.float32) / 255.0
        depth_metric = depth_norm * scale

        # Quality filter — skip low variance frames
        if depth_metric.mean() < 0.1 or depth_metric.std() < 0.01:
            skipped_quality += 1
            continue

        # Clamp to outdoor range
        depth_metric = np.clip(depth_metric, 0.0, 80.0)

        # Save as 16-bit PNG in millimeters
        depth_mm  = (depth_metric * 1000).astype(np.uint16)
        out_path  = output_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), depth_mm)
        saved += 1

    print(f"\nDone.")
    print(f"  Saved:           {saved}")
    print(f"  Skipped (water): {skipped_water}")
    print(f"  Skipped (quality): {skipped_quality}")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth-dir',    type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/depth_maps'))
    parser.add_argument('--raw-dir',      type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/raw_frames'))
    parser.add_argument('--output-dir',   type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/metric_labels'))
    parser.add_argument('--scale-factor', type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/scale_factor.txt'))
    args = parser.parse_args()
    scale_to_metric(args)
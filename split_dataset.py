import os
import random
import shutil
from pathlib import Path
import argparse


def split_dataset(args):
    raw_dir    = Path(args.raw_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    # Get all frames that have matching labels
    raw_frames = sorted(raw_dir.glob('*.png'))
    paired = []
    for raw in raw_frames:
        label = labels_dir / raw.name
        if label.exists():
            paired.append(raw.stem)

    print(f"Found {len(paired)} paired frames")

    # Separate by clip to avoid data leakage
    clip_1913 = [f for f in paired if '1913' in f]
    clip_1916 = [f for f in paired if '1916' in f]
    print(f"  Clip 1913: {len(clip_1913)} frames")
    print(f"  Clip 1916: {len(clip_1916)} frames")

    # Shuffle each clip independently
    random.seed(42)
    random.shuffle(clip_1913)
    random.shuffle(clip_1916)

    def split_clip(frames, train=0.8, val=0.1):
        n = len(frames)
        t = int(n * train)
        v = int(n * val)
        return frames[:t], frames[t:t+v], frames[t+v:]

    train_1913, val_1913, test_1913 = split_clip(clip_1913)
    train_1916, val_1916, test_1916 = split_clip(clip_1916)

    train = train_1913 + train_1916
    val   = val_1913   + val_1916
    test  = test_1913  + test_1916

    print(f"\nSplit: train={len(train)} val={len(val)} test={len(test)}")

    # Create directory structure
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'rgb').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'depth').mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_split(stems, split_name):
        for stem in stems:
            src_rgb   = raw_dir    / f"{stem}.png"
            src_depth = labels_dir / f"{stem}.png"
            dst_rgb   = output_dir / split_name / 'rgb'   / f"{stem}.png"
            dst_depth = output_dir / split_name / 'depth' / f"{stem}.png"
            shutil.copy2(src_rgb,   dst_rgb)
            shutil.copy2(src_depth, dst_depth)
        print(f"  {split_name}: copied {len(stems)} pairs")

    copy_split(train, 'train')
    copy_split(val,   'val')
    copy_split(test,  'test')

    print(f"\nDataset saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir',    type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/raw_frames'))
    parser.add_argument('--labels-dir', type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/metric_labels'))
    parser.add_argument('--output-dir', type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/dataset'))
    args = parser.parse_args()
    split_dataset(args)
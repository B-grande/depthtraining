import cv2
import numpy as np
import argparse
import os
from pathlib import Path

# Global state
clicked_value = None
real_distance_m = 0.66  # bow distance in meters

def mouse_callback(event, x, y, flags, param):
    global clicked_value
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_img = param['depth']
        clicked_value = depth_img[y, x]
        print(f"Clicked pixel ({x}, {y}) — depth value: {clicked_value}")
        print(f"Scale factor: {real_distance_m / (clicked_value / 255.0):.4f}")

def find_scale_factor(args):
    """Interactive tool to find scale factor by clicking the bow."""
    global clicked_value

    # Load a sample frame and its depth map
    raw_frames = sorted(Path(args.raw_dir).glob('*.png'))
    depth_maps = sorted(Path(args.depth_dir).glob('*.png'))

    if not raw_frames:
        print("No frames found")
        return

    print(f"Found {len(raw_frames)} frames")
    print("Instructions:")
    print("  N/P: next/previous frame")
    print("  Left click on BOW TIP to get scale factor")
    print("  S: save scale factor and exit")
    print("  Q: quit without saving")

    idx = 0
    scale_factor = None

    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)

    while True:
        raw_path = raw_frames[idx]
        # Find matching depth map
        stem = raw_path.stem
        depth_path = Path(args.depth_dir) / f"{stem}.png"

        raw = cv2.imread(str(raw_path))
        if depth_path.exists():
            depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        else:
            depth = np.zeros((raw.shape[0], raw.shape[1]), dtype=np.uint8)

        param = {'depth': depth}
        cv2.setMouseCallback('Calibration', mouse_callback, param)

        # Display side by side
        depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
        display = np.hstack([raw, depth_colored])

        if clicked_value is not None:
            depth_norm = clicked_value / 255.0
            if depth_norm > 0:
                scale_factor = real_distance_m / depth_norm
                label = f"Scale: {scale_factor:.4f} (bow={clicked_value}, real={real_distance_m}m)"
                cv2.putText(display, label, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.putText(display, f"Frame {idx+1}/{len(raw_frames)}: {stem}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "N/P: next/prev | Click bow tip | S: save | Q: quit",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Calibration', display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = min(idx + 1, len(raw_frames) - 1)
            clicked_value = None
        elif key == ord('p'):
            idx = max(idx - 1, 0)
            clicked_value = None
        elif key == ord('s') and scale_factor is not None:
            print(f"\nSaving scale factor: {scale_factor:.4f}")
            with open(args.output, 'w') as f:
                f.write(f"{scale_factor:.6f}\n")
            print(f"Saved to {args.output}")
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir',   type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/raw_frames'))
    parser.add_argument('--depth-dir', type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/depth_maps'))
    parser.add_argument('--output',    type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/scale_factor.txt'))
    args = parser.parse_args()
    find_scale_factor(args)
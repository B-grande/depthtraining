import rerun as rr
import rerun.blueprint as rrb
import cv2
import numpy as np
import argparse
import os


def load_raw_frames(raw_path):
    if raw_path and os.path.exists(raw_path):
        print("Loading raw metric depth...")
        data = np.load(raw_path, mmap_mode='r')
        print(f"Loaded {len(data)} frames")
        return data
    return None


def run(args):
    all_videos = [args.original] + args.depth
    caps = [cv2.VideoCapture(v) for v in all_videos]

    orig_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    raw_frames = load_raw_frames(args.raw)

    # Initialize rerun
    rr.init("depth_viewer", spawn=True)

    # Build blueprint — one panel per video
    panel_contents = []
    panel_contents.append(rrb.Spatial2DView(name="Original", origin="/original"))
    for i, depth_path in enumerate(args.depth):
        name = os.path.basename(os.path.dirname(depth_path))
        panel_contents.append(rrb.Spatial2DView(name=name, origin=f"/depth/{i}"))

    if raw_frames is not None:
        panel_contents.append(rrb.Spatial2DView(name="Metric Depth", origin="/metric"))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(*panel_contents),
        collapse_panels=True
    )
    rr.send_blueprint(blueprint)

    print(f"Processing {total_frames} frames...")

    frame_idx = 0
    while True:
        rets = []
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            rets.append(ret)
            frames.append(frame)

        if not all(rets):
            break

        # Set timeline
        rr.set_time("time", duration=frame_idx / fps)
        rr.set_time("frame", sequence=frame_idx)

        # Log original
        orig_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
        rr.log("original", rr.Image(orig_rgb))

        # Log depth videos
        for i, frame in enumerate(frames[1:]):
            depth_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rr.log(f"depth/{i}", rr.Image(depth_rgb))

        # Log metric depth as actual depth image
        if raw_frames is not None and frame_idx < len(raw_frames):
            metric = raw_frames[frame_idx]
            rr.log("metric", rr.DepthImage(metric, meter=1.0))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Logged frame {frame_idx}/{total_frames}")

    for cap in caps:
        cap.release()

    print("Done. Rerun viewer should be open.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True)
    parser.add_argument('--depth', type=str, nargs='+', required=True)
    parser.add_argument('--raw', type=str, default=None,
                        help='Path to raw metric .npy file')
    args = parser.parse_args()
    run(args)
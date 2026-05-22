import cv2
import numpy as np
import argparse
import os
from collections import deque

click_points = []
paused = False
roi = None
roi_drawing = False
roi_start = None
depth_history = deque(maxlen=150)


def mouse_callback(event, x, y, flags, param):
    global roi, roi_drawing, roi_start

    cols = param['cols']
    orig_w = param['orig_w']
    orig_h = param['orig_h']

    panel_col = x // orig_w
    panel_row = y // orig_h
    panel_idx = panel_row * cols + panel_col
    sample_x = x % orig_w
    sample_y = y % orig_h

    if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_SHIFTKEY):
        roi_drawing = True
        roi_start = (x, y)
        roi = None

    elif event == cv2.EVENT_MOUSEMOVE and roi_drawing:
        param['roi_preview'] = (roi_start, (x, y))

    elif event == cv2.EVENT_LBUTTONUP and roi_drawing:
        roi_drawing = False
        x1, y1 = roi_start
        x2, y2 = x, y
        x1 = max(0, min(x1 % orig_w, orig_w - 1))
        x2 = max(0, min(x2 % orig_w, orig_w - 1))
        y1 = max(0, min(y1 % orig_h, orig_h - 1))
        y2 = max(0, min(y2 % orig_h, orig_h - 1))
        roi = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        param['roi_preview'] = None
        depth_history.clear()

    elif event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_SHIFTKEY):
        if param['raw_frames'] is not None and panel_idx > 0:
            frame_idx = param['frame_idx']
            raw = param['raw_frames']
            if frame_idx < len(raw):
                raw_frame = raw[frame_idx]
                sy = min(sample_y, raw_frame.shape[0] - 1)
                sx = min(sample_x, raw_frame.shape[1] - 1)
                value = raw_frame[sy, sx]
                label = f"{value:.2f}m"
            else:
                label = "N/A"
        elif param['depth_frame'] is not None:
            depth_frame = param['depth_frame']
            sx = min(sample_x, depth_frame.shape[1] - 1)
            sy = min(sample_y, depth_frame.shape[0] - 1)
            value = depth_frame[sy, sx]
            label = f"{value} ({value/255*100:.1f}%)"
        else:
            return
        click_points.append((x, y, label))


def draw_header(display, labels, orig_w, orig_h, cols):
    for i, label in enumerate(labels):
        row = i // cols
        col = i % cols
        name = os.path.basename(os.path.dirname(label)) if i > 0 else "Original"
        x = col * orig_w + 10
        y = row * orig_h + 50
        cv2.putText(display, name, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)


def draw_roi(display, orig_w, orig_h, cols, num_panels):
    if roi is None:
        return
    x1, y1, x2, y2 = roi
    for i in range(num_panels):
        row = i // cols
        col = i % cols
        ox = col * orig_w
        oy = row * orig_h
        cv2.rectangle(display,
                      (ox + x1, oy + y1),
                      (ox + x2, oy + y2),
                      (0, 255, 255), 2 if i == 0 else 1)
        if i == 0:
            cv2.putText(display, "ROI", (ox + x1, max(oy + y1 - 10, oy + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)


def draw_colorbar(display, orig_h, orig_w, cols, rows, max_depth=20.0):
    bar_w = 25
    bar_h = orig_h // 2
    bar_x = cols * orig_w - bar_w - 20
    bar_y = orig_h // 4
    steps = 30
    seg_h = max(1, bar_h // steps)
    for i in range(steps):
        ratio = 1.0 - (i / steps)
        val = int(ratio * 255)
        color_bgr = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
                                      cv2.COLORMAP_INFERNO)[0][0]
        cv2.rectangle(display,
                      (bar_x, bar_y + i * seg_h),
                      (bar_x + bar_w, bar_y + (i + 1) * seg_h),
                      color_bgr.tolist(), -1)
    cv2.putText(display, f"{max_depth:.0f}m", (bar_x - 55, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(display, "0m", (bar_x - 35, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_depth_graph(display, total_h, total_w):
    if len(depth_history) < 2:
        return

    graph_h = 180
    graph_w = min(800, total_w - 40)
    graph_x = 20
    graph_y = total_h - graph_h - 60

    overlay = display.copy()
    cv2.rectangle(overlay, (graph_x - 10, graph_y - 10),
                  (graph_x + graph_w + 10, graph_y + graph_h + 10),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)
    cv2.rectangle(display, (graph_x - 10, graph_y - 10),
                  (graph_x + graph_w + 10, graph_y + graph_h + 10),
                  (80, 80, 80), 2)

    for i in range(1, 4):
        gy = graph_y + int(i * graph_h / 4)
        cv2.line(display, (graph_x, gy), (graph_x + graph_w, gy), (50, 50, 50), 1)

    cv2.putText(display, "ROI Avg Depth",
                (graph_x + 5, graph_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    values = list(depth_history)
    min_v, max_v = min(values), max(values)
    range_v = max(max_v - min_v, 0.01)

    pts = []
    for i, v in enumerate(values):
        px = graph_x + int(i / max(len(values) - 1, 1) * graph_w)
        py = graph_y + graph_h - int((v - min_v) / range_v * (graph_h - 40)) - 10
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(display, pts[i-1], pts[i], (0, 220, 255), 2)

    current = values[-1]
    label = f"{current:.2f}m" if isinstance(current, float) else f"{current:.1f}"
    cv2.putText(display, f"Current: {label}",
                (graph_x + 5, graph_y + graph_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    if len(values) > 5:
        rate = (values[-1] - values[-6]) / 5
        color = (0, 80, 255) if rate < 0 else (0, 255, 80)
        direction = "approaching" if rate < 0 else "receding"
        cv2.putText(display, f"Rate: {rate:.3f} ({direction})",
                    (graph_x + 380, graph_y + graph_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    cv2.putText(display, f"max: {max_v:.2f}",
                (graph_x + graph_w - 160, graph_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(display, f"min: {min_v:.2f}",
                (graph_x + graph_w - 160, graph_y + graph_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)


def get_roi_depth(depth_frame, raw_frames, frame_idx, is_metric):
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    if is_metric and raw_frames is not None and frame_idx < len(raw_frames):
        raw = raw_frames[frame_idx]
        region = raw[y1:y2, x1:x2]
        if region.size == 0:
            return None
        return float(np.mean(region))
    elif depth_frame is not None:
        region = depth_frame[y1:y2, x1:x2]
        if region.size == 0:
            return None
        return float(np.mean(region))
    return None


def build_grid(frames, cols):
    num = len(frames)
    rows = (num + cols - 1) // cols
    h, w = frames[0].shape[:2]

    # Pad with black frames if needed
    while len(frames) < rows * cols:
        frames.append(np.zeros_like(frames[0]))

    row_imgs = []
    for r in range(rows):
        row = np.hstack(frames[r * cols:(r + 1) * cols])
        row_imgs.append(row)
    return np.vstack(row_imgs)


def run(args):
    global paused

    all_videos = [args.original] + args.depth
    caps = [cv2.VideoCapture(v) for v in all_videos]
    num_panels = len(caps)
    cols = args.cols

    orig_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = (num_panels + cols - 1) // cols
    total_w = orig_w * cols
    total_h = orig_h * rows

    raw_frames = None
    if args.raw:
        print("Loading raw metric depth...")
        raw_frames = np.load(args.raw, mmap_mode='r')
        print(f"Loaded {len(raw_frames)} frames")

    cv2.namedWindow('Depth Viewer', cv2.WINDOW_NORMAL)
    param = {
        'depth_frame': None,
        'orig_w': orig_w,
        'orig_h': orig_h,
        'cols': cols,
        'raw_frames': raw_frames,
        'frame_idx': 0,
        'roi_preview': None,
    }
    cv2.setMouseCallback('Depth Viewer', mouse_callback, param)

    frames = [None] * num_panels
    frame_idx = 0
    is_metric = args.raw is not None

    while True:
        if not paused:
            rets, new_frames = [], []
            for cap in caps:
                ret, frame = cap.read()
                rets.append(ret)
                new_frames.append(frame)

            if not all(rets):
                for cap in caps:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                click_points.clear()
                depth_history.clear()
                frame_idx = 0
                param['frame_idx'] = 0
                continue

            frames = new_frames
            frame_idx += 1
            param['frame_idx'] = frame_idx

            if frames[1] is not None:
                param['depth_frame'] = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)

        if any(f is None for f in frames):
            continue

        roi_val = get_roi_depth(param['depth_frame'], raw_frames, frame_idx, is_metric)
        if roi_val is not None:
            depth_history.append(roi_val)

        display = build_grid(list(frames), cols)

        for (x, y, label) in click_points:
            cv2.circle(display, (x, y), 7, (0, 255, 0), -1)
            cv2.putText(display, label, (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        draw_header(display, all_videos, orig_w, orig_h, cols)
        draw_roi(display, orig_w, orig_h, cols, num_panels)
        draw_depth_graph(display, total_h, total_w)

        if args.raw:
            draw_colorbar(display, orig_h, orig_w, cols, rows)

        if param['roi_preview']:
            p1, p2 = param['roi_preview']
            cv2.rectangle(display, p1, p2, (0, 255, 255), 1)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(display,
                    f"{status} | SPACE: play/pause | F: step | C: clear | R: clear ROI | Shift+drag: ROI | Q: quit",
                    (10, total_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Frame {frame_idx}",
                    (total_w - 200, total_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow('Depth Viewer', display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('f') and paused:
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if ret:
                    frames[i] = frame
            frame_idx += 1
            param['frame_idx'] = frame_idx
            if frames[1] is not None:
                param['depth_frame'] = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        elif key == ord('c'):
            click_points.clear()
        elif key == ord('r'):
            roi = None
            depth_history.clear()

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True)
    parser.add_argument('--depth', type=str, nargs='+', required=True)
    parser.add_argument('--raw', type=str, default=None)
    parser.add_argument('--cols', type=int, default=3,
                        help='Number of columns in grid layout')
    args = parser.parse_args()
    run(args)
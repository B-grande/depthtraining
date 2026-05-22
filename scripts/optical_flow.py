import cv2
import numpy as np
import argparse
from collections import deque

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

FEATURE_PARAMS = dict(
    maxCorners=150,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

MOTION_THRESHOLD = [0.5]
OUTLIER_MULT = [3.0]
motion_history = deque(maxlen=90)


def detect_horizon(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100, minLineLength=gray.shape[1] // 3,
                             maxLineGap=50)
    if lines is None:
        return gray.shape[0] // 2

    horizon_candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        mid_y = (y1 + y2) / 2
        if angle < 15 and mid_y < gray.shape[0] * 0.7:
            horizon_candidates.append(int(mid_y))

    if not horizon_candidates:
        return gray.shape[0] // 3

    return int(np.median(horizon_candidates))


def build_water_mask(frame, horizon_y, padding=30):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    h = frame.shape[0]
    mask[min(horizon_y + padding, h - 1):, :] = 255
    return mask


def is_wave_motion(dx, dy, mag, avg_mag):
    if mag < MOTION_THRESHOLD[0]:
        return False
    if mag > avg_mag * OUTLIER_MULT[0]:
        return False
    return True


def draw_motion_graph(display, orig_h, orig_w):
    if len(motion_history) < 2:
        return

    graph_h = 180
    graph_w = min(800, orig_w - 40)
    graph_x = 20
    graph_y = orig_h - graph_h - 60

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

    cv2.putText(display, "Wave Motion Magnitude (water region only)",
                (graph_x + 5, graph_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    values = list(motion_history)
    max_v = max(max(values), 1)

    pts = []
    for i, v in enumerate(values):
        px = graph_x + int(i / max(len(values) - 1, 1) * graph_w)
        py = graph_y + graph_h - int(v / max_v * (graph_h - 40)) - 10
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(display, pts[i-1], pts[i], (0, 220, 255), 2)

    current = values[-1]
    if len(values) > 5:
        rate = (values[-1] - values[-6]) / 5
        direction = "WAVE INCOMING" if rate > 0.5 else \
                    "RECEDING" if rate < -0.5 else "STABLE"
        color = (0, 255, 0) if rate > 0.5 else \
                (0, 0, 255) if rate < -0.5 else (255, 255, 0)
        cv2.putText(display, f"{direction}  mag: {current:.2f}  rate: {rate:.3f}",
                    (graph_x + 5, graph_y + graph_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def run(args):
    cap = cv2.VideoCapture(args.video)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to open video")
        return

    orig_h, orig_w = prev_frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    horizon_y = detect_horizon(prev_gray)
    water_mask = build_water_mask(prev_frame, horizon_y)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=water_mask, **FEATURE_PARAMS)
    trail_mask = np.zeros_like(prev_frame)
    paused = False
    frame_count = 0
    show_horizon = True

    cv2.namedWindow('Optical Flow', cv2.WINDOW_NORMAL)

    def process_frame(frame):
        nonlocal prev_gray, p0, trail_mask, frame_count, horizon_y, water_mask

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 60 == 0:
            horizon_y = detect_horizon(gray)
            water_mask = build_water_mask(frame, horizon_y)

        wave_magnitudes = []

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, p0, None, **LK_PARAMS
            )

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                mags = [np.sqrt((n[0]-o[0])**2 + (n[1]-o[1])**2)
                        for n, o in zip(good_new, good_old)]
                avg_mag = np.mean(mags) if mags else 1.0

                for new, old, mag in zip(good_new, good_old, mags):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    dx, dy = a - c, b - d

                    bi, ai = int(b), int(a)
                    if bi >= orig_h or ai >= orig_w or bi < 0 or ai < 0:
                        continue
                    if water_mask[bi, ai] == 0:
                        continue
                    if not is_wave_motion(dx, dy, mag, avg_mag):
                        continue

                    wave_magnitudes.append(mag)
                    color = (0, 255, 0) if dy > 0 else (0, 0, 255)
                    trail_mask = cv2.line(trail_mask, (int(a), int(b)),
                                          (int(c), int(d)), color, 2)
                    display = cv2.circle(display, (int(a), int(b)), 4, color, -1)

                p0 = good_new.reshape(-1, 1, 2)

        if wave_magnitudes:
            motion_history.append(np.mean(wave_magnitudes))
        elif motion_history:
            motion_history.append(0.0)

        frame_count += 1
        if frame_count % 30 == 0:
            p0 = cv2.goodFeaturesToTrack(gray, mask=water_mask, **FEATURE_PARAMS)
            trail_mask = np.zeros_like(frame)

        prev_gray = gray
        combined = cv2.add(display, trail_mask)

        if show_horizon:
            cv2.line(combined, (0, horizon_y), (orig_w, horizon_y), (255, 100, 0), 2)
            cv2.putText(combined, f"horizon y={horizon_y}", (10, max(horizon_y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2, cv2.LINE_AA)

        overlay = combined.copy()
        cv2.rectangle(overlay, (0, 0), (orig_w, max(horizon_y, 0)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, combined, 0.6, 0, combined)

        draw_motion_graph(combined, orig_h, orig_w)

        status = "PAUSED" if paused else "PLAYING"
        cv2.putText(combined,
                    f"{status} | SPACE: pause | F: step | H: horizon | W/S: adjust | [/]: thresh {MOTION_THRESHOLD[0]} | ,/.: outlier {OUTLIER_MULT[0]} | Q: quit",
                    (10, orig_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(combined, f"Frame {frame_count}",
                    (orig_w - 200, orig_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        return combined

    combined = process_frame(prev_frame)
    cv2.imshow('Optical Flow', combined)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                horizon_y = detect_horizon(prev_gray)
                water_mask = build_water_mask(frame, horizon_y)
                p0 = cv2.goodFeaturesToTrack(prev_gray, mask=water_mask, **FEATURE_PARAMS)
                trail_mask = np.zeros_like(frame)
                frame_count = 0
                motion_history.clear()

            combined = process_frame(frame)
            cv2.imshow('Optical Flow', combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('f') and paused:
            ret, frame = cap.read()
            if ret:
                combined = process_frame(frame)
                cv2.imshow('Optical Flow', combined)
        elif key == ord('h'):
            show_horizon = not show_horizon
        elif key == ord('w'):
            horizon_y = max(0, horizon_y - 10)
            water_mask = build_water_mask(prev_frame, horizon_y)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=water_mask, **FEATURE_PARAMS)
            trail_mask = np.zeros_like(prev_frame)
        elif key == ord('s'):
            horizon_y = min(orig_h - 1, horizon_y + 10)
            water_mask = build_water_mask(prev_frame, horizon_y)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=water_mask, **FEATURE_PARAMS)
            trail_mask = np.zeros_like(prev_frame)
        elif key == ord(']'):
            MOTION_THRESHOLD[0] = round(min(5.0, MOTION_THRESHOLD[0] + 0.1), 1)
        elif key == ord('['):
            MOTION_THRESHOLD[0] = round(max(0.1, MOTION_THRESHOLD[0] - 0.1), 1)
        elif key == ord('.'):
            OUTLIER_MULT[0] = round(min(10.0, OUTLIER_MULT[0] + 0.5), 1)
        elif key == ord(','):
            OUTLIER_MULT[0] = round(max(1.0, OUTLIER_MULT[0] - 0.5), 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    args = parser.parse_args()
    run(args)
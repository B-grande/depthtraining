import dearpygui.dearpygui as dpg
import cv2
import numpy as np
import argparse
import os
import threading

state = {
    'frame_idx': 0,
    'paused': True,
    'caps': [],
    'raw_frames': None,
    'orig_w': 0,
    'orig_h': 0,
    'total_w': 0,
    'num_panels': 0,
    'click_points': [],
    'labels': [],
    'running': True,
    'frames': [],
    'needs_update': False,
}


def read_next_frames():
    rets, frames = [], []
    for cap in state['caps']:
        ret, frame = cap.read()
        rets.append(ret)
        frames.append(frame)
    if not all(rets):
        for cap in state['caps']:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        state['frame_idx'] = 0
        state['click_points'].clear()
        return False
    state['frames'] = frames
    state['frame_idx'] += 1
    return True


def push_texture():
    frames = state['frames']
    if not frames or any(f is None for f in frames):
        return
    combined = np.hstack(frames)
    rgba = cv2.cvtColor(combined, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
    dpg.set_value("video_texture", rgba.flatten().tolist())


def refresh_overlay():
    if not dpg.does_item_exist("overlay_layer"):
        return
    dpg.delete_item("overlay_layer", children_only=True)

    orig_w = state['orig_w']
    orig_h = state['orig_h']

    # Panel labels
    for i, label in enumerate(state['labels']):
        name = os.path.basename(os.path.dirname(label)) if i > 0 else "Original"
        dpg.draw_text((i * orig_w + 10, 10), name, size=22,
                      color=(255, 255, 100, 255), parent="overlay_layer")

    # Click points
    for (x, y, label) in state['click_points']:
        dpg.draw_circle((x, y), 7, color=(0, 255, 0, 255),
                        fill=(0, 255, 0, 180), parent="overlay_layer")
        dpg.draw_text((x + 12, y - 10), label, size=18,
                      color=(0, 255, 0, 255), parent="overlay_layer")

    # Colorbar
    if state['raw_frames'] is not None:
        total_w = state['total_w']
        bar_x = total_w - 55
        bar_y = orig_h // 4
        bar_h = orig_h // 2
        steps = 30
        seg_h = max(1, bar_h // steps)
        for i in range(steps):
            ratio = 1.0 - (i / steps)
            val = int(ratio * 255)
            bgr = cv2.applyColorMap(np.array([[val]], dtype=np.uint8),
                                    cv2.COLORMAP_INFERNO)[0][0]
            color = (int(bgr[2]), int(bgr[1]), int(bgr[0]), 255)
            dpg.draw_rectangle((bar_x, bar_y + i * seg_h),
                                (bar_x + 20, bar_y + (i + 1) * seg_h),
                                color=color, fill=color, parent="overlay_layer")
        dpg.draw_text((bar_x - 38, bar_y - 2), "20m", size=15,
                      color=(255, 255, 255, 255), parent="overlay_layer")
        dpg.draw_text((bar_x - 28, bar_y + bar_h), "0m", size=15,
                      color=(255, 255, 255, 255), parent="overlay_layer")

    # Frame counter
    dpg.draw_text((10, orig_h - 28), f"Frame {state['frame_idx']}",
                  size=15, color=(180, 180, 180, 255), parent="overlay_layer")


def on_canvas_click(sender, app_data):
    mouse_pos = dpg.get_drawing_mouse_pos()
    x, y = int(mouse_pos[0]), int(mouse_pos[1])
    orig_w = state['orig_w']
    orig_h = state['orig_h']

    if x < 0 or y < 0 or y >= orig_h or x >= state['total_w']:
        return

    panel_idx = x // orig_w
    sample_x = x % orig_w

    if state['raw_frames'] is not None and panel_idx > 0:
        fidx = min(state['frame_idx'], len(state['raw_frames']) - 1)
        raw = state['raw_frames'][fidx]
        sy = min(y, raw.shape[0] - 1)
        sx = min(sample_x, raw.shape[1] - 1)
        value = raw[sy, sx]
        label = f"{value:.2f}m"
    else:
        frames = state['frames']
        if panel_idx < len(frames) and frames[panel_idx] is not None:
            gray = cv2.cvtColor(frames[panel_idx], cv2.COLOR_BGR2GRAY)
            sy = min(y, gray.shape[0] - 1)
            sx = min(sample_x, gray.shape[1] - 1)
            value = gray[sy, sx]
            label = f"{value} ({value/255*100:.1f}%)"
        else:
            return

    state['click_points'].append((x, y, label))
    refresh_overlay()


def video_thread():
    import time
    while state['running']:
        if not state['paused']:
            read_next_frames()
            state['needs_update'] = True
        time.sleep(1 / 30)


def run(args):
    all_videos = [args.original] + args.depth
    state['labels'] = all_videos
    state['caps'] = [cv2.VideoCapture(v) for v in all_videos]
    state['num_panels'] = len(all_videos)

    orig_w = int(state['caps'][0].get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(state['caps'][0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_w = orig_w * len(all_videos)
    state['orig_w'] = orig_w
    state['orig_h'] = orig_h
    state['total_w'] = total_w
    state['frames'] = [None] * len(all_videos)

    if args.raw:
        print("Loading raw metric depth...")
        state['raw_frames'] = np.load(args.raw, mmap_mode='r')
        print(f"Loaded {len(state['raw_frames'])} frames")

    # Read first frames
    for i, cap in enumerate(state['caps']):
        ret, frame = cap.read()
        if ret:
            state['frames'][i] = frame
    state['frame_idx'] = 1

    dpg.create_context()
    dpg.create_viewport(title='Depth Viewer', width=total_w + 20, height=orig_h + 100)
    dpg.setup_dearpygui()

    # Init texture
    combined = np.hstack(state['frames'])
    rgba = cv2.cvtColor(combined, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0

    with dpg.texture_registry():
        dpg.add_raw_texture(total_w, orig_h, rgba.flatten().tolist(),
                            tag="video_texture", format=dpg.mvFormat_Float_rgba)

    with dpg.window(label="Depth Viewer", tag="main_window", no_close=True,
                    width=total_w + 20, height=orig_h + 100):

        with dpg.group(horizontal=True):
            dpg.add_button(label="  Play  ",
                           callback=lambda: state.update({'paused': False}))
            dpg.add_button(label="  Pause  ",
                           callback=lambda: state.update({'paused': True}))
            dpg.add_button(label="  Step Frame  ", callback=lambda: [
                read_next_frames(),
                push_texture(),
                refresh_overlay()
            ])
            dpg.add_button(label="  Clear Points  ", callback=lambda: [
                state['click_points'].clear(), refresh_overlay()
            ])

        dpg.add_spacer(height=5)

        with dpg.drawlist(width=total_w, height=orig_h, tag="canvas"):
            dpg.draw_image("video_texture", (0, 0), (total_w, orig_h), tag="video_image")
            with dpg.draw_layer(tag="overlay_layer"):
                pass

        with dpg.item_handler_registry(tag="canvas_handler"):
            dpg.add_item_clicked_handler(callback=on_canvas_click)
        dpg.bind_item_handler_registry("canvas", "canvas_handler")

    refresh_overlay()

    thread = threading.Thread(target=video_thread, daemon=True)
    thread.start()

    dpg.set_primary_window("main_window", True)
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        if state['needs_update']:
            push_texture()
            refresh_overlay()
            state['needs_update'] = False
        dpg.render_dearpygui_frame()

    state['running'] = False
    for cap in state['caps']:
        cap.release()
    dpg.destroy_context()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True)
    parser.add_argument('--depth', type=str, nargs='+', required=True)
    parser.add_argument('--raw', type=str, default=None)
    args = parser.parse_args()
    run(args)
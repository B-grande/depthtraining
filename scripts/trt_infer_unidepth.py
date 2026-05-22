import argparse
import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt


def preprocess_frame(frame, input_h, input_w):
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (input_w, input_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb  = (rgb - mean) / std
    rgb  = rgb.transpose(2, 0, 1)
    return rgb[np.newaxis].astype(np.float32), orig_h, orig_w


def process_frame(frame, context, h_input, h_pts3d, h_conf,
                  d_input, d_pts3d, d_conf, d_intrinsics, stream,
                  input_shape, pts3d_shape, conf_shape):

    input_h = input_shape[2]
    input_w = input_shape[3]

    input_data, orig_h, orig_w = preprocess_frame(frame, input_h, input_w)
    np.copyto(h_input, input_data.ravel())

    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.set_tensor_address('rgbs', int(d_input))
    context.set_tensor_address('pts_3d', int(d_pts3d))
    context.set_tensor_address('confidence', int(d_conf))
    context.set_tensor_address('intrinsics', int(d_intrinsics))
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_pts3d, d_pts3d, stream)
    cuda.memcpy_dtoh_async(h_conf, d_conf, stream)
    stream.synchronize()

    # pts_3d is shape [1, 3, H, W] — channel 2 is Z (depth)
    pts3d = np.reshape(h_pts3d, pts3d_shape)
    conf  = np.reshape(h_conf,  conf_shape)
    pts3d = np.nan_to_num(pts3d, nan=0.0, posinf=0.0, neginf=0.0)
    conf  = np.nan_to_num(conf,  nan=0.0, posinf=0.0, neginf=0.0)

    # Extract Z channel for depth
    depth = pts3d[0, 2, :, :]  # [H, W]
    conf  = conf[0, 0, :, :]   # [H, W]

    raw_depth = cv2.resize(depth, (orig_w, orig_h))
    raw_conf  = cv2.resize(conf,  (orig_w, orig_h))

    # Visualize depth
    max_depth = 20.0
    depth_vis = np.clip(depth / max_depth, 0, 1) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.resize(depth_vis, (orig_w, orig_h))
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

    # Visualize confidence
    conf_norm = cv2.normalize(conf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    conf_norm = cv2.resize(conf_norm, (orig_w, orig_h))
    conf_colored = cv2.applyColorMap(conf_norm, cv2.COLORMAP_VIRIDIS)

    return depth_colored, conf_colored, orig_w, orig_h, raw_depth, raw_conf


def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        input_shape    = context.get_tensor_shape('rgbs')
        pts3d_shape    = context.get_tensor_shape('pts_3d')
        conf_shape     = context.get_tensor_shape('confidence')
        intrinsics_shape = context.get_tensor_shape('intrinsics')

        h_input  = cuda.pagelocked_empty(trt.volume(input_shape),    dtype=np.float32)
        h_pts3d  = cuda.pagelocked_empty(trt.volume(pts3d_shape),    dtype=np.float32)
        h_conf   = cuda.pagelocked_empty(trt.volume(conf_shape),     dtype=np.float32)
        h_intrinsics = cuda.pagelocked_empty(trt.volume(intrinsics_shape), dtype=np.float32)

        d_input      = cuda.mem_alloc(h_input.nbytes)
        d_pts3d      = cuda.mem_alloc(h_pts3d.nbytes)
        d_conf       = cuda.mem_alloc(h_conf.nbytes)
        d_intrinsics = cuda.mem_alloc(h_intrinsics.nbytes)
        stream       = cuda.Stream()

        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        depth_writer = None
        conf_writer  = None
        raw_mmap     = None
        frame_idx    = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            depth_colored, conf_colored, orig_w, orig_h, raw_depth, raw_conf = process_frame(
                frame, context, h_input, h_pts3d, h_conf,
                d_input, d_pts3d, d_conf, d_intrinsics, stream,
                input_shape, pts3d_shape, conf_shape
            )

            if depth_writer is None:
                video_name = os.path.splitext(os.path.basename(args.video))[0]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                depth_path = os.path.join(args.outdir, f"{video_name}_depth.mp4")
                conf_path  = os.path.join(args.outdir, f"{video_name}_conf.mp4")
                raw_path   = os.path.join(args.outdir, f"{video_name}_raw.npy")
                depth_writer = cv2.VideoWriter(depth_path, fourcc, fps, (orig_w, orig_h))
                conf_writer  = cv2.VideoWriter(conf_path,  fourcc, fps, (orig_w, orig_h))
                raw_mmap = np.lib.format.open_memmap(
                    raw_path, mode='w+', dtype=np.float32,
                    shape=(total_frames, orig_h, orig_w)
                )

            depth_writer.write(depth_colored)
            conf_writer.write(conf_colored)
            raw_mmap[frame_idx] = raw_depth

            frame_idx += 1
            print(f"Processing frame {frame_idx}/{total_frames}", end='\r')

        cap.release()
        if depth_writer:
            depth_writer.release()
        if conf_writer:
            conf_writer.release()
        if raw_mmap is not None:
            del raw_mmap

        print(f"\nDone.")
        print(f"  Depth:      {depth_path}")
        print(f"  Confidence: {conf_path}")
        print(f"  Raw depth:  {raw_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',  type=str, required=True)
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='./output_unidepth')
    args = parser.parse_args()
    run(args)
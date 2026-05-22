import argparse
import os
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from depth_anything.util.transform import load_image


def process_frame(frame, context, h_input, h_output, d_input, d_output, stream, input_shape, output_shape, metric=False):
    tmp_path = "/tmp/tmp_frame.jpg"
    cv2.imwrite(tmp_path, frame)
    input_image, (orig_h, orig_w) = load_image(tmp_path)

    np.copyto(h_input, input_image.ravel())
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.set_tensor_address('input', int(d_input))
    context.set_tensor_address('output', int(d_output))
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    depth = np.reshape(h_output, output_shape[1:]) if len(output_shape) == 3 else np.reshape(h_output, output_shape[2:])
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    if metric:
        raw_depth = depth.copy()
        max_depth = 20.0
        depth_vis = np.clip(depth / max_depth, 0, 1) * 255.0
    else:
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 0:
            depth_vis = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_vis = np.zeros_like(depth)
        raw_depth = None

    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.resize(depth_vis, (orig_w, orig_h))

    if metric:
        raw_depth = cv2.resize(raw_depth, (orig_w, orig_h))

    return depth_vis, orig_w, orig_h, raw_depth


def run(args):
    os.makedirs(args.outdir, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, 'rb') as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        input_shape = context.get_tensor_shape('input')
        output_shape = context.get_tensor_shape('output')
        h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
        h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
        d_input = cuda.mem_alloc(h_input.nbytes)
        d_output = cuda.mem_alloc(h_output.nbytes)
        stream = cuda.Stream()

        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_writer = None
        raw_mmap = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            depth_vis, orig_w, orig_h, raw_depth = process_frame(
                frame, context, h_input, h_output,
                d_input, d_output, stream, input_shape, output_shape,
                metric=args.metric
            )

            out_frame = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_INFERNO)

            if out_writer is None:
                video_name = os.path.splitext(os.path.basename(args.video))[0]
                out_path = os.path.join(args.outdir, f"{video_name}_depth.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

                if args.metric:
                    raw_path = os.path.join(args.outdir, f"{video_name}_depth_raw.npy")
                    raw_mmap = np.lib.format.open_memmap(
                        raw_path, mode='w+', dtype=np.float32,
                        shape=(total_frames, orig_h, orig_w)
                    )

            out_writer.write(out_frame)

            if args.metric and raw_depth is not None and raw_mmap is not None:
                raw_mmap[frame_idx] = raw_depth

            frame_idx += 1
            print(f"Processing frame {frame_idx}/{total_frames}", end='\r')

        cap.release()
        out_writer.release()

        if args.metric and raw_mmap is not None:
            del raw_mmap

        print(f"\nDone. Saved to {out_path}")
        if args.metric:
            print(f"Raw metric depth saved to {raw_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run depth estimation on video with TensorRT.')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--engine', type=str, required=True)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--metric', action='store_true', help='Enable metric depth output in meters')
    args = parser.parse_args()
    run(args)
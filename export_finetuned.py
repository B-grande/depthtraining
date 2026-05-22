import os
import sys
import torch
import argparse

sys.path.insert(0, os.path.expanduser('~/depth_anything_V2/Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',    type=str, default='vitl',
                        choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned .pth file')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.expanduser('~/depth_anything_V2/training_data/checkpoints'))
    args = parser.parse_args()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    print(f"Loading fine-tuned model: {args.checkpoint}")
    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to('cpu').eval()

    dummy = torch.ones((1, 3, args.input_size, args.input_size))
    onnx_path = os.path.join(args.output_dir,
                             f'da2_maritime_{args.encoder}.onnx')

    import onnxscript  # noqa
    torch.onnx.export(model, dummy, onnx_path, opset_version=18,
                      input_names=['input'], output_names=['output'])
    print(f"ONNX exported: {onnx_path}")

    engine_path = onnx_path.replace('.onnx', '.engine')
    print(f"Building TRT engine: {engine_path}")
    os.system(f"trtexec --onnx={onnx_path} --saveEngine={engine_path} --fp16")
    print(f"Done: {engine_path}")


if __name__ == '__main__':
    main()
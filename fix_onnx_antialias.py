import onnx
import sys

def fix_antialias(input_path, output_path):
    model = onnx.load(input_path)
    fixed = 0
    for node in model.graph.node:
        if node.op_type == "Resize":
            for attr in node.attribute:
                if attr.name == "antialias" and attr.i == 1:
                    attr.i = 0
                    fixed += 1
    print(f"Fixed {fixed} antialias attributes")
    onnx.save(model, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    fix_antialias(sys.argv[1], sys.argv[2])
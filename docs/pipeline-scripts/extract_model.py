import onnx
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract specific parts of an ONNX model.")
    parser.add_argument("--model_input", required=True, help="Path to the input ONNX model.")
    parser.add_argument("--model_output", required=True, help="Path to save the extracted ONNX model.")
    parser.add_argument("--node_input", required=True, nargs="+", help="List of input node.")
    parser.add_argument("--node_output", required=True, nargs="+", help="List of output node.")
    args = parser.parse_args()

    onnx.utils.extract_model(args.model_input, args.model_output, args.node_input, args.node_output)
    print(f"Extracted model saved to {args.model_output}")

if __name__ == "__main__":
    main()

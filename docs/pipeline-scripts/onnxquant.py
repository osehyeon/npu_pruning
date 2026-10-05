import os
import cv2
import numpy as np
import argparse
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat

def load_data(dp, calib_img_cnt, sz=(608, 608)):
    data = []
    cnt = 0
    for f in os.listdir(dp):
        if cnt >= calib_img_cnt:
            break
        p = os.path.join(dp, f)
        if os.path.isfile(p) and p.lower().endswith((".png", ".jpg", ".jpeg")):
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, sz)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            data.append(img)
            cnt += 1
    return data

class CalibrationDataReaderImpl(CalibrationDataReader):
    def __init__(self, dataset_path, calib_img_cnt):
        self.data = load_data(dataset_path, calib_img_cnt)
        self.iterator = iter(self.data)

    def get_next(self):
        try:
            data = next(self.iterator)
            return {args.input_name: data}
        except StopIteration:
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize an ONNX model to UINT8")
    parser.add_argument("--input-model", required=True, help="Path to input ONNX model")
    parser.add_argument("--output-model", required=True, help="Path to save quantized ONNX model")
    parser.add_argument("--calib-dataset", required=True, help="Path to calibration dataset directory")
    parser.add_argument("--calib-count", type=int, default=1, help="Number of calibration images to use")
    parser.add_argument("--input-name", required=True, help="Name of the model's input node")
    args = parser.parse_args()

    calib_data_reader = CalibrationDataReaderImpl(args.calib_dataset, args.calib_count)

    quantize_static(
        model_input=args.input_model,
        model_output=args.output_model,
        calibration_data_reader=calib_data_reader,
        quant_format=QuantFormat.QOperator,
        per_channel=False,
        weight_type=QuantType.QUInt8,
        activation_type=QuantType.QUInt8
    )

    print(f"Quantized model saved to: {args.output_model}")
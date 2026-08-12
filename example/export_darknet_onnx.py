#!/usr/bin/env python3
"""ultralytics archive 의 Darknet 모델을 **원시 헤드 출력** ONNX 로 내보낸다.

기존 평가 경로(`decoders.decode_std` + `nms_per_class`)를 그대로 쓰려면 출력이
`(1, 255, H, W)` 세 개여야 한다. archive 의 `YOLOLayer` 는 `training=True` 일 때
`(1, 3, H, W, 85)` 를 반환하므로 축을 되돌려 준다.

BN 은 eval 로 두고 `YOLOLayer.training` 만 켜서 디코딩·NMS 가 그래프에 안 들어가게 한다.

    python export_darknet_onnx.py --cfg x.cfg --weights x.weights --out x.onnx
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn


class RawHeads(nn.Module):
    """(1,3,H,W,85) 세 개 → (1,255,H,W) 세 개."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        ys = self.model(x)
        if isinstance(ys, tuple):
            ys = ys[0]
        outs = []
        for y in ys:
            b, na, ny, nx, no = y.shape
            outs.append(y.permute(0, 1, 4, 2, 3).reshape(b, na * no, ny, nx))
        return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=608)
    ap.add_argument("--opset", type=int, default=12)
    args = ap.parse_args()

    from models import Darknet, load_darknet_weights, YOLOLayer

    model = Darknet(args.cfg, img_size=(args.size, args.size))
    if args.weights.endswith(".pt"):
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
    else:
        load_darknet_weights(model, args.weights)

    # Darknet.forward_once 와 YOLOLayer 가 모두 training 분기를 타야 원시 출력이 나온다.
    # 대신 BatchNorm 만 eval 로 되돌려 러닝 통계를 쓰게 한다.
    model.train()
    n_bn = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            n_bn += 1
    n_yolo = sum(1 for m in model.modules() if isinstance(m, YOLOLayer))
    print(f"YOLOLayer {n_yolo}개 원시 출력 모드 · BatchNorm {n_bn}개 추론 모드")

    wrapper = RawHeads(model)
    x = torch.zeros(1, 3, args.size, args.size)
    with torch.no_grad():
        outs = wrapper(x)
    print("출력 형태:", [tuple(o.shape) for o in outs])

    torch.onnx.export(
        wrapper, x, args.out, opset_version=args.opset,
        input_names=["images"],
        output_names=[f"head{i}" for i in range(len(outs))],
        do_constant_folding=True,
    )
    print(f"→ {args.out}")

    import onnx
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print("ONNX 검증 통과 · 노드", len(m.graph.node))


if __name__ == "__main__":
    main()

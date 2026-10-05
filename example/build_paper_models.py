#!/usr/bin/env python3
"""논문용 모델 세트를 `paper/model/` 로 모으고 매니페스트를 만든다.

프루닝율 × staircase 격자와 그 파생물을 한자리에 놓고, 각 모델의
해시·구조·연산량·NPU 실측을 하나의 JSON 으로 고정한다.

    python build_paper_models.py [--link]      # --link 면 복사 대신 심볼릭 링크
"""
from __future__ import annotations

import argparse, hashlib, json, math, shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

WS = Path.home() / "workspace_docs"
DRIVE = WS / "YOLOv3_Antara_재현패키지/전처리/구글드라이브"
MODEL = WS / "model"
OUT = WS / "paper/model"
# 논문에 쓰지 않은 재현본은 2026-09-16 에 여기로 옮겼다. 없으면 건너뛴다
ARCHIVE = WS.parent.parent / "archive/neubla/models-not-in-paper"
ALIGN = 64

# 연계 문서 E02 — 2025.03.25 NPU latency 스윕 (초)
NPU_LAT = {(0, None): 0.0571,
           (10, 8): 0.0562, (10, 16): 0.0561, (10, 32): 0.0561, (10, 64): 0.0558,
           (30, 8): 0.0524, (30, 16): 0.0505, (30, 32): 0.0466, (30, 64): 0.0474,
           (50, 8): 0.0462, (50, 16): 0.0463, (50, 32): 0.0463, (50, 64): 0.0413}

FILENAME = {"original": "yolov3-12_bn-unfolded.onnx"}

ITEMS = [("original", 0, None, "parent", DRIVE / "yolov3-12.onnx",
          "ONNX Model Zoo yolov3-12 (프루닝 부모, BN 미폴드). 다운로드: https://github.com/onnx/models/blob/main/validated/vision/object_detection_segmentation/yolov3/README.md")]
for r in (10, 30, 50):
    for sc in (8, 16, 32, 64):
        ITEMS.append((f"sweep_r{r}_sc{sc}", r, sc, "sweep",
                      DRIVE / f"sweep_12종/{r}/compressed_{sc}_yolov3-12_sim.onnx",
                      "2025-03 스윕 · 재학습 전"))
for r in (30, 40, 45):
    ITEMS.append((f"comp250514_r{r}_sc64", r, 64, "comp250514",
                  DRIVE / f"250514_comp/{r}_compressed_64_yolov3-12_sim.onnx",
                  "2025-05-14 · 재학습 전"))
# 2026-09-16 · model/ 의 사본을 지워 final/ 이 유일한 실물이다. 출처를 자기 자신으로 두면
# 아래 복사 단계는 건너뛰고 통계와 지문만 다시 계산한다.
ITEMS += [
    ("final_r40_sc64", 40, 64, "final", OUT / "final" / "final_r40_sc64.onnx",
     "Nota 제출본 · 재학습 후"),
    ("final_r50_sc64", 50, 64, "final", OUT / "final" / "final_r50_sc64.onnx",
     "Nota 50% · 재학습 후"),
    ("ours_r40_sc64_ep10", 40, 64, "ours", ARCHIVE / "paper-model-ours" / "ours_r40_sc64_ep10.onnx",
     "본 연구 재현: 마스크 복원 프루닝 + trainvalno5k 10 epoch 재학습 · 논문 미사용. archive 에 있음"),
]


def stats(path, size=608):
    m = onnx.load(str(path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    params = int(sum(a.size for a in init.values()))
    hw, raw, pad, chans = size, 0, 0, []
    for n in m.graph.node:
        if n.op_type == "Conv" and len(n.input) > 1 and n.input[1] in init:
            w = init[n.input[1]]
            st = next((a.ints[0] for a in n.attribute if a.name == "strides"), 1)
            hw //= st
            o, c, kh, kw = w.shape
            raw += o * c * kh * kw * hw * hw
            pad += (math.ceil(o/ALIGN)*ALIGN) * (math.ceil(c/ALIGN)*ALIGN) * kh * kw * hw * hw
            chans.append(int(o))
        elif n.op_type in ("Resize", "Upsample"):
            hw *= 2
    del m, init
    return dict(params=params, macs_raw=int(raw), macs_padded=int(pad),
                channels_total=int(sum(chans)), n_conv=len(chans),
                n_conv_aligned64=int(sum(1 for c in chans if c % ALIGN == 0)))


def sha256(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while (b := f.read(buf)):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--link", action="store_true", help="복사 대신 심볼릭 링크")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    manifest, base = [], None
    for name, r, sc, group, src, note in ITEMS:
        if not src.exists():
            print(f"  건너뜀 (없음): {name}"); continue
        d = OUT / group; d.mkdir(exist_ok=True)
        # parent 는 BN 을 접지 않은 Model Zoo 원본임을 파일명에 남긴다. 접은 판본은 model/yolov3_original_fp32.onnx
        dst = d / FILENAME.get(name, f"{name}.onnx")
        if not dst.exists():
            if args.link:
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)
        st = stats(src)
        if base is None:
            base = st
        rec = dict(name=name, group=group, ratio_nominal=r, staircase=sc,
                   file=str(dst.relative_to(WS)), size_bytes=src.stat().st_size,
                   sha256=sha256(src), note=note, **st)
        rec["macs_raw_ratio"] = round(st["macs_raw"] / base["macs_raw"], 4)
        rec["macs_padded_ratio"] = round(st["macs_padded"] / base["macs_padded"], 4)
        rec["pad_waste_pct"] = round(100 * (st["macs_padded"] / st["macs_raw"] - 1), 2)
        rec["channels_reduction_pct"] = round(
            100 * (1 - st["channels_total"] / base["channels_total"]), 2)
        rec["npu_latency_s"] = NPU_LAT.get((r, sc))
        manifest.append(rec)
        print(f"  {name:<22} {src.stat().st_size/1e6:7.1f} MB  "
              f"채널 {st['channels_total']:>6}  MACs {st['macs_raw']/1e9:5.1f}G  "
              f"패딩낭비 {rec['pad_waste_pct']:5.1f}%")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n모델 {len(manifest)}개 · 매니페스트 → {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()

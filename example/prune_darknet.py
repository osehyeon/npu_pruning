#!/usr/bin/env python3
"""Darknet YOLOv3 구조적 프루닝 — cfg + weights 를 채널 마스크로 잘라낸다.

용도: 원본 `yolov3.weights` 에 프루닝을 적용해 재학습 가능한 형태(cfg + weights)로
내보낸다. 산출물은 ultralytics archive 의 `train.py` / `test.py` 가 그대로 먹는다.

마스크 출처 두 가지:
  --masks <json>   복원한 Nota 마스크 (재학습 전 실물에서 오차 0 으로 역복원)
  --ratio <float>  L2 상위 k + 64 정렬 규칙으로 직접 산출 (residual 그룹 공유)

darknet .weights 레이아웃 (cfg 의 conv 등장 순서대로):
    BN 有 : biases[n] scales[n] mean[n] var[n] weights[n*c*k*k]
    BN 無 : biases[n] weights[n*c*k*k]
가중치는 (out, in, h, w) 순.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np


# ---------- cfg 파싱 ----------
def parse_cfg(path):
    blocks, cur = [], None
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("["):
            if cur is not None:
                blocks.append(cur)
            cur = {"__type__": line[1:-1].strip()}
        else:
            k, v = line.split("=", 1)
            cur[k.strip()] = v.strip()
    if cur is not None:
        blocks.append(cur)
    return blocks


def write_cfg(blocks, path):
    out = []
    for b in blocks:
        out.append(f"[{b['__type__']}]")
        for k, v in b.items():
            if k != "__type__":
                out.append(f"{k}={v}")
        out.append("")
    Path(path).write_text("\n".join(out) + "\n")


# ---------- 마스크 전파 ----------
def propagate(blocks, conv_masks, in_ch0=3):
    """레이어별 출력 채널 마스크를 계산. conv_masks[i]=None 이면 미프루닝."""
    layer_mask = []          # darknet 레이어 인덱스(=[net] 제외) 순
    ci = -1
    for b in blocks:
        t = b["__type__"]
        if t == "net":
            continue
        if t == "convolutional":
            ci += 1
            n = int(b["filters"])
            m = conv_masks.get(ci)
            layer_mask.append(np.arange(n) if m is None else np.asarray(sorted(m)))
        elif t == "shortcut":
            frm = int(b["from"])
            a, c = layer_mask[-1], layer_mask[frm]
            if len(a) != len(c):
                raise ValueError(f"shortcut 채널 불일치: {len(a)} vs {len(c)}")
            layer_mask.append(a)
        elif t == "route":
            idxs = [int(x) for x in b["layers"].split(",")]
            idxs = [i if i < 0 else i - len(layer_mask) for i in idxs]   # 절대→상대
            parts = [layer_mask[i] for i in idxs]
            layer_mask.append(np.concatenate([np.arange(len(p)) for p in parts]))
        elif t == "upsample":
            layer_mask.append(layer_mask[-1])
        elif t == "yolo":
            layer_mask.append(layer_mask[-1])
        else:
            layer_mask.append(layer_mask[-1])
    return layer_mask


def conv_in_mask(blocks, layer_mask, conv_layer_idx):
    """conv 가 받는 입력 채널 인덱스 (직전 레이어의 출력 마스크)."""
    if conv_layer_idx == 0:
        return np.arange(3)
    prev = layer_mask[conv_layer_idx - 1]
    return prev


# ---------- L2 규칙으로 마스크 생성 ----------
def masks_by_l2(convs_w, groups, keep_counts):
    """convs_w: darknet 순 conv 가중치. groups: 공유 마스크 그룹(리스트의 리스트)."""
    masks = {}
    grouped = {i for g in groups for i in g}
    for i, w in enumerate(convs_w):
        k = keep_counts.get(i)
        if k is None or k >= w.shape[0] or i in grouped:
            continue
        l2 = np.linalg.norm(w.reshape(w.shape[0], -1), axis=1)
        masks[i] = sorted(np.argsort(-l2)[:k].tolist())
    for g in groups:
        k = keep_counts.get(g[0])
        if k is None or k >= convs_w[g[0]].shape[0]:
            continue
        agg = np.zeros(convs_w[g[0]].shape[0])
        for i in g:
            agg += np.linalg.norm(convs_w[i].reshape(convs_w[i].shape[0], -1), axis=1)
        sel = sorted(np.argsort(-agg)[:k].tolist())
        for i in g:
            masks[i] = sel
    return masks


# ---------- 가중치 입출력 ----------
def load_weights(path, blocks):
    raw = np.fromfile(path, dtype=np.uint8)
    hdr = raw[:20].copy()
    buf = np.frombuffer(raw[20:].tobytes(), dtype=np.float32)
    p, layers = 0, []
    ci = -1
    prev_ch = 3
    for b in blocks:
        if b["__type__"] != "convolutional":
            continue
        ci += 1
        n = int(b["filters"]); k = int(b["size"])
        bn = int(b.get("batch_normalize", 0))
        c = b["__in_ch__"]
        d = {"idx": ci, "bn": bn}
        d["biases"] = buf[p:p + n]; p += n
        if bn:
            d["scales"] = buf[p:p + n]; p += n
            d["mean"] = buf[p:p + n]; p += n
            d["var"] = buf[p:p + n]; p += n
        d["w"] = buf[p:p + n * c * k * k].reshape(n, c, k, k); p += n * c * k * k
        layers.append(d)
    assert p == buf.size, f"파일 소비 불일치 {p} vs {buf.size}"
    return hdr, layers


def annotate_in_channels(blocks):
    """각 conv 블록에 입력 채널 수(__in_ch__)를 기록 (원본 cfg 기준)."""
    ch = [3]
    for b in blocks:
        t = b["__type__"]
        if t == "net":
            continue
        if t == "convolutional":
            b["__in_ch__"] = ch[-1]
            ch.append(int(b["filters"]))
        elif t == "shortcut":
            ch.append(ch[-1])
        elif t == "route":
            idxs = [int(x) for x in b["layers"].split(",")]
            n = len(ch) - 1
            tot = 0
            for i in idxs:
                j = i if i < 0 else i - n
                tot += ch[j]
            ch.append(tot)
        else:
            ch.append(ch[-1])
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--masks", help="prune_masks.json")
    ap.add_argument("--tag", default="40", help="masks json 안의 키")
    ap.add_argument("--out-cfg", required=True)
    ap.add_argument("--out-weights", required=True)
    args = ap.parse_args()

    blocks = annotate_in_channels(parse_cfg(args.cfg))
    hdr, layers = load_weights(args.weights, blocks)
    print(f"cfg conv {len(layers)}개 · 가중치 로드 완료")

    md = json.load(open(args.masks))[args.tag]["masks_darknet_order"]
    conv_masks = {int(k): v for k, v in md.items()}
    n_pruned = sum(1 for v in conv_masks.values() if v is not None)
    print(f"마스크 {args.tag}% : 프루닝 레이어 {n_pruned}개")

    layer_mask = propagate(blocks, conv_masks)

    # conv 블록의 레이어 인덱스 목록
    conv_layer_idx, li = [], -1
    for b in blocks:
        if b["__type__"] == "net":
            continue
        li += 1
        if b["__type__"] == "convolutional":
            conv_layer_idx.append(li)

    new_bufs, tot_in, tot_out = [], 0, 0
    for d, lidx in zip(layers, conv_layer_idx):
        om = layer_mask[lidx]
        im = conv_in_mask(blocks, layer_mask, lidx)
        w = d["w"][np.ix_(om, im)]
        tot_in += d["w"].size; tot_out += w.size
        new_bufs.append(d["biases"][om])
        if d["bn"]:
            new_bufs += [d["scales"][om], d["mean"][om], d["var"][om]]
        new_bufs.append(w.ravel())

    # cfg 갱신
    ci = -1
    for b in blocks:
        if b["__type__"] != "convolutional":
            continue
        ci += 1
        b["filters"] = str(len(layer_mask[conv_layer_idx[ci]]))
        b.pop("__in_ch__", None)
    write_cfg(blocks, args.out_cfg)

    out = np.concatenate([np.asarray(x, np.float32).ravel() for x in new_bufs])
    with open(args.out_weights, "wb") as f:
        f.write(hdr.tobytes()); f.write(out.tobytes())

    print(f"가중치 원소 {tot_in:,} → {tot_out:,} (−{100*(1-tot_out/tot_in):.2f}%)")
    print(f"→ {args.out_cfg}")
    print(f"→ {args.out_weights}  ({Path(args.out_weights).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

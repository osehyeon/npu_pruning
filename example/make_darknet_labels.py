#!/usr/bin/env python3
"""COCO2014 어노테이션 → darknet 형식 라벨 일괄 생성.

ultralytics archive 의 데이터로더는 이미지 경로의 `images` 를 `labels` 로 바꾼 위치에서
`<stem>.txt` 를 찾는다. 한 줄에 객체 하나:

    <class 0..79> <x_center> <y_center> <width> <height>      (모두 0~1 정규화)

train2014 + val2014 전량(123,287장)을 한 번에 만들어 두면 어떤 부분집합 목록이든
그대로 쓸 수 있다. iscrowd 는 학습에서 제외한다.

    python make_darknet_labels.py --root ~/yolov3_prune/data
"""
from __future__ import annotations

import argparse, json
from pathlib import Path


def build(root: Path, split: str):
    ann = root / f"annotations/instances_{split}.json"
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)

    d = json.load(open(ann))
    cat_ids = sorted(c["id"] for c in d["categories"])
    cid2idx = {c: i for i, c in enumerate(cat_ids)}
    imgs = {i["id"]: i for i in d["images"]}

    per = {}
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        per.setdefault(a["image_id"], []).append((cid2idx[a["category_id"]], x, y, w, h))

    n_lbl, n_obj, n_empty, n_miss = 0, 0, 0, 0
    for iid, info in imgs.items():
        if not (img_dir / info["file_name"]).exists():
            n_miss += 1
            continue
        lines = []
        W, H = info["width"], info["height"]
        for c, x, y, w, h in per.get(iid, []):
            lines.append(f"{c} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
        (lbl_dir / (Path(info["file_name"]).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        n_lbl += 1
        n_obj += len(lines)
        n_empty += (not lines)
    print(f"[{split}] 라벨 {n_lbl:,}개 · 객체 {n_obj:,} · 빈 라벨 {n_empty:,} · 이미지 없음 {n_miss:,}")
    return n_lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train2014", "val2014"])
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    total = sum(build(root, s) for s in args.splits)
    print(f"합계 {total:,}개")


if __name__ == "__main__":
    main()

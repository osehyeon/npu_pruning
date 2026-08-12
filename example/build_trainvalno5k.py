#!/usr/bin/env python3
"""Darknet 원본과 동일한 학습/평가 분할을 만든다 (COCO 2014 기준).

    trainvalno5k = train2014 + val2014 − 5k     (118,287장)  ← 원본이 학습한 집합
    5k                                          (  5,000장)  ← 원본이 held-out 한 집합

원본과 **같은 데이터**로 재학습해야 프루닝 손실만 분리됩니다. 학습 데이터가 다르면
그 차이가 프루닝 손실로 잘못 읽힙니다.

    python build_trainvalno5k.py --root <coco2014> --fivek 5k.part --out-dir <dir>
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fivek", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root, out = Path(args.root).expanduser(), Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    ids5k = {int(re.search(r"(\d+)\.jpg", l).group(1))
             for l in open(args.fivek) if ".jpg" in l}
    print(f"5k 목록 {len(ids5k):,}장")

    train, ev = [], []
    for split in ("train2014", "val2014"):
        d = json.load(open(root / f"annotations/instances_{split}.json"))
        n_t = n_e = 0
        for i in d["images"]:
            p = root / "images" / split / i["file_name"]
            if not p.exists():
                continue
            if i["id"] in ids5k:
                ev.append(str(p)); n_e += 1
            else:
                train.append(str(p)); n_t += 1
        print(f"  {split}: 학습 {n_t:,} · 평가 {n_e:,}")

    train.sort(); ev.sort()
    (out / "trainvalno5k.txt").write_text("\n".join(train) + "\n")
    (out / "eval5k.txt").write_text("\n".join(ev) + "\n")

    print(f"\n학습 trainvalno5k {len(train):,}장   (공칭 118,287 일치: {len(train) == 118287})")
    print(f"평가 5k           {len(ev):,}장")
    inter = set(train) & set(ev)
    print(f"교집합 {len(inter)}장  {'✓' if not inter else '✗ 오염!'}")


if __name__ == "__main__":
    main()

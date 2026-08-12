#!/usr/bin/env python3
"""오염 없는 학습/평가 분할을 만든다.

문제 — 두 계열의 held-out 집합이 어긋나 있다.

    원본(Darknet)   학습 trainvalno5k = train2014 + val2014 − 5k(pjreddie)
    Nota 프루닝본    학습 train2017    = train2014 + val2014 − val2017
    5k ∩ val2017 = 643장뿐

해법 — 우리 재학습은 **5k 와 val2017 을 모두 제외**하고 돌린다.

    학습 = (train2014 ∪ val2014) − (5k ∪ val2017)
    평가 = 5k 5,000장     ← 원본에도, 우리 재학습본에도 깨끗

그러면 643장이 아니라 5,000장 전수로 편향 없는 비교가 된다.

    python build_clean_split.py --root <coco2014 루트> --out-dir <출력>
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path


def load_ann(p):
    d = json.load(open(p))
    return {i["id"]: i for i in d["images"]}, d["annotations"], d["categories"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="coco2014 루트 (images/ annotations/ 포함)")
    ap.add_argument("--fivek", required=True, help="pjreddie 5k.part")
    ap.add_argument("--val2017-ann", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ids_5k = {int(re.search(r"(\d+)\.jpg", l).group(1))
              for l in open(args.fivek) if ".jpg" in l}
    ids_v17 = {i["id"] for i in json.load(open(args.val2017_ann))["images"]}
    hold = ids_5k | ids_v17
    print(f"5k {len(ids_5k)} · val2017 {len(ids_v17)} · 합집합(제외 대상) {len(hold)} "
          f"· 교집합 {len(ids_5k & ids_v17)}")

    train_imgs, _, _ = load_ann(root / "annotations/instances_train2014.json")
    val_imgs, _, _ = load_ann(root / "annotations/instances_val2014.json")
    print(f"train2014 {len(train_imgs)} · val2014 {len(val_imgs)}")

    lines, n_skip = [], 0
    for imgs, sub in ((train_imgs, "train2014"), (val_imgs, "val2014")):
        for iid, info in imgs.items():
            if iid in hold:
                n_skip += 1
                continue
            p = root / "images" / sub / info["file_name"]
            if p.exists():
                lines.append(str(p))
    lines.sort()
    (out / "train_clean.txt").write_text("\n".join(lines) + "\n")
    print(f"\n학습셋 {len(lines):,}장  (제외 {n_skip:,}장)")
    print(f"  참고: train2017 은 118,287장 → 여기서 5k 4,357장을 더 뺀 규모")
    print(f"→ {out/'train_clean.txt'}")

    # 평가셋 (5k) 목록도 같이
    ev = []
    for iid in sorted(ids_5k):
        info = val_imgs.get(iid)
        if info:
            p = root / "images/val2014" / info["file_name"]
            if p.exists():
                ev.append(str(p))
    (out / "eval_5k.txt").write_text("\n".join(ev) + "\n")
    print(f"평가셋 {len(ev):,}장 → {out/'eval_5k.txt'}")


if __name__ == "__main__":
    main()

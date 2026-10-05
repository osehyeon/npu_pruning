#!/bin/bash
export PYTHONPATH=$HOME/demo_1127
cd /home/osehn/workspace_docs/example
export COCO_IMG=/home/osehn/datasets/coco2014/val2014
export COCO_ANN=/home/osehn/datasets/coco2014/annotations/instances_val2014.json
export COCO_IDS=/home/osehn/datasets/coco2014/ids_5k.txt
for T in R_orig p40_egshalf; do
  echo "=== $T $(date '+%H:%M:%S') ==="
  ./.venv38/bin/python eval_map_npu.py $T --limit 0
done
echo "NPU5KDONE $(date '+%H:%M:%S')"

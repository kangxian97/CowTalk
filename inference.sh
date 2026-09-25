#!/bin/bash
# Run volume inference. Set COW_DATA_ROOT and COWTALK_CHECKPOINT first.
cd "$(dirname "$0")"

DATA_ROOT="${COW_DATA_ROOT:-data}"
FOLD="${FOLD:-1}"
CHECKPOINT="${COWTALK_CHECKPOINT:-checkpoints/checkpoint_best.pth}"
OUT_DIR="${COWTALK_OUT:-results/fold_${FOLD}}"

python inference_volume.py \
    --checkpoint "$CHECKPOINT" \
    --pointcloud-data-root "$DATA_ROOT/pointclouds" \
    --split-file "$DATA_ROOT/splits/fold_${FOLD}.json" \
    --instructions-dir "$DATA_ROOT/instructions_generated" \
    --correction-result-dir "$OUT_DIR" \
    --fold "$FOLD" \
    --input-points 10000 \
    --query-points 8000 \
    --input-fg-ratio 0.7 \
    --input-random-ratio 0.3 \
    --query-fg-ratio 0.9 \
    --query-random-ratio 0.1 \
    --batch-size 10000

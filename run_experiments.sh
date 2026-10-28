#!/usr/bin/env bash
# Full experiment run: stage 1 (architectures) -> pick best -> stage 2 (hyperparameters).
cd "$(dirname "$0")"
PY="${PYTHON:-python}"   # override with PYTHON=/path/to/python ./run_experiments.sh
$PY gen_configs.py stage1
for c in configs/s1_*.yaml; do echo "=== $c $(date +%H:%M)"; $PY train.py --config "$c"; done
$PY gen_configs.py stage2
for c in configs/s2_*.yaml; do echo "=== $c $(date +%H:%M)"; $PY train.py --config "$c"; done
echo "=== ALL DONE $(date)"

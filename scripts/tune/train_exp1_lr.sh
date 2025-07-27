#!/bin/bash
set -a
source .env
set +a

python src/tune/run_tuner.py \
  --name ray_exp1 \
  --config_file configs/tune/search_lr.yaml \
  --num_samples 30 \
  --max_num_epochs 20 \
  --min_num_epochs 5 \
  --train_data_path "$TRAIN_DATA_PATH" \
  --gpus_per_trial 1 \
  --storage_path "$STORAGE_PATH" \
  --logging_path logs/logging/tune/ray_exp1.log \
  --logging_basic_level DEBUG \
  --logging_console_level INFO \
  --logging_file_level INFO \
  --save_path outputs/hparam
#!/bin/bash

# スクリプトのディレクトリに移動 (この行はオプション)
cd "$(dirname "$0")"

# 必要ならスクリプトの実行環境を明示 (仮想環境は不要と記載あり)
# 例: module load python3などのSIF環境設定があれば記述

# プロジェクトディレクトリに移動
cd Projects

# Pythonスクリプトを実行
python image_compression.py \
  IMAGE_PATH=data/Multilayer_para3_64.npy \
  IMAGE_DTYPE=ndarray \
  NUM_EPOCH=100000 \
  FP_G0_BIT=2 \
  FEATURE_PYRAMID_G0_CHANNEL=4 \
  FP_G1_BIT=2 \
  FEATURE_PYRAMID_G1_CHANNEL=6 \
  FEATURE_PYRAMID_SIZE_RATE=4 \
  COMPRESSION_METHOD=2 \
  IMAGE_DIMENSION=3 \
  IMAGE_SIZE=512 \
  CROP_MIP_LEVEL=8

# 実行後にシェルを保持する場合
$SHELL

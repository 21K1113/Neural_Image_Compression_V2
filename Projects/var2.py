import torch
import sys
import os
from utils import *

over_write_variable_dict = {
    "COMPRESSION_METHOD": "int",
    "RANDOM_SEED": "int",

    "NUM_EPOCH": "int",
    "IMAGE_SIZE": "int",
    "IMAGE_3D_SIZE": "int",
    "MAX_MIP_LEVEL": "int",

    "FP_G0_BIT": "int",
    "FEATURE_PYRAMID_G0_CHANNEL": "int",
    "FEATURE_PYRAMID_SIZE_RATE": "int",
    "FP_G1_BIT": "int",
    "FEATURE_PYRAMID_G1_CHANNEL": "int",

    "IMAGE_PATH": "str",
    "PROJECT_NAME": "str",
    "IMAGE_DTYPE": "str",

    "MLP_NUM_DTYPE": "int",
    "UNIFORM_DISTRIBUTION_RATE": "float",
    "IMAGE_DIMENSION": "int",
    "IMAGE_BIT": "int",
    "OUTPUT_BIT": "int",
    "PE_CHANNEL": "int",
    "HIDDEN_LAYER_CHANNEL": "int",
    "CROP_MIP_LEVEL": "int",
    "NUM_CROP": "int",

    "INTERVAL_PRINT": "int",
    "INTERVAL_SAVE_MODEL": "int",
    "INTERVAL_WRITER_LOSS": "int",
    "INTERVAL_WRITER_PSNR": "int",
    "INTERVAL_WRITER_TIME": "int",

    "TF_NO_MIP": "bool",
    "TF_USE_TRI_PE": "bool",
    "TF_TRAIN_MODEL": "bool",
    "TF_DECORD_ALL": "bool",
    "TF_SHOW_RESULT": "bool",
    "TF_PRINT_LOSS": "bool",
    "TF_PRINT_PSNR": "bool",
    "TF_PRINT_TIME": "bool",
    "TF_WRITE_PSNR": "bool",
    "TF_WRITE_TIME": "bool",
}

# 圧縮する画像のパス
# IMAGE_PATH = 'data/misty_64_64.avi'
# IMAGE_PATH = 'data/Multilayer_para3_64.npy'
IMAGE_PATH = 'data/sancho_512.png'

# 保存用の固有名詞
PROJECT_NAME = "image_compression"

# 入力画像のデータ型
# IMAGE_DTYPE = "movie"
# IMAGE_DTYPE = "ndarray"
IMAGE_DTYPE = "image"

COMPRESSION_METHOD = 1
# 1: 2次元圧縮
# 2: 3次元を2次元に平坦化
# 3: 特徴ピラミッドを3次元に拡張
# 4: 提案手法

# 全体のbit数
MLP_NUM_DTYPE = 32

RANDOM_SEED = 1234                  # ランダムのシード値

NUM_EPOCH = 1000                    # 学習回数
UNIFORM_DISTRIBUTION_RATE = 0.05    # 一様分布からサンプリングする割合
IMAGE_3D_SIZE = 64                  # 平坦化時のみ使用する
IMAGE_SIZE = 512                    # 入力画像サイズ
IMAGE_DIMENSION = 2                 # 入力画像次元
MAX_MIP_LEVEL = 9                   # 入力画像サイズのミップレベル
IMAGE_BIT = 8                       # 入力画像のbit数（不完全）
OUTPUT_BIT = 8                      # 出力画像のbit数（不完全）
PE_CHANNEL = 6                      # 位置エンコーディングの次元数
HIDDEN_LAYER_CHANNEL = 64           # デコーダの中間層のノード数

FEATURE_PYRAMID_G0_CHANNEL = 12     # 特徴ピラミッドのG0のチャンネル数
FP_G0_BIT = 8                       # 特徴ピラミッドのG0の量子化ビット数
FEATURE_PYRAMID_SIZE_RATE = 2       # 特徴ピラミッドのG0のビットサイズ比率（2なら1/4になる）
FEATURE_PYRAMID_G1_CHANNEL = 12     # 特徴ピラミッドのG1のチャンネル数
FP_G1_BIT = 8                       # 特徴ピラミッドのG0の量子化ビット数

CROP_MIP_LEVEL = 8                  # ランダムクロップのクロップサイズ
NUM_CROP = 8                        # ランダムクロップの数（＝バッチ数）

INTERVAL_PRINT = 100                # logやPSNRのprintをする間隔
INTERVAL_WRITER_LOSS = 100          # lossのwriteをする間隔
INTERVAL_WRITER_PSNR = 100          # lossのwriteをする間隔
INTERVAL_WRITER_TIME = 100          # lossのwriteをする間隔
INTERVAL_SAVE_MODEL = 100000        # 学習途中のモデルを保存する間隔

TF_NO_MIP = True                    # mipmapを生成しない
TF_USE_TRI_PE = True                # 三角関数ではなく、三角波に基づいた位置エンコーディングを使用する（もう使ってない）
TF_TRAIN_MODEL = True               # 学習するかどうか（Falseのとき、学習済みモデルをロードする）
TF_DECORD_ALL = True               # 1画素ずつデコードする

TF_SHOW_RESULT = False              # 結果をmatplotlibかなんかで表示
TF_PRINT_LOSS = True                # 一定間隔でlossをprintする
TF_PRINT_PSNR = True                # 一定間隔でpsnrをprintする
TF_PRINT_TIME = True                # 一定間隔で時間をprintする
TF_WRITE_PSNR = True                # Tensorboardに学習ステップごとのPSNRを記録する
TF_WRITE_TIME = True                # Tensorboardに学習ステップごとの時間を記録する

# コマンドラインから変数を取得し値を更新
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        for var in over_write_variable_dict.keys():
            if arg.startswith(var + "="):
                value = judge_value(arg, over_write_variable_dict[var], var)
                exec(f"{var} = {value}", globals())

print(IMAGE_PATH)

# デバイスの設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASENAME = os.path.basename(IMAGE_PATH)
IMAGE_EXT = os.path.splitext(IMAGE_PATH)[1][1:]
IMAGE_DTYPE = dtype_from_ext(IMAGE_EXT)
FEATURE_PYRAMID_SIZE = IMAGE_SIZE // pow(2, FEATURE_PYRAMID_SIZE_RATE)      # 特徴ピラミッドのサイズ
FP_DIMENSION = IMAGE_DIMENSION
if COMPRESSION_METHOD == 2:
    FP_DIMENSION = 2



if TF_NO_MIP:
    MAX_MIP_LEVEL = 0
DECODER_INPUT_CHANNEL = FEATURE_PYRAMID_G0_CHANNEL * pow(2, FP_DIMENSION) + FEATURE_PYRAMID_G1_CHANNEL + PE_CHANNEL * FP_DIMENSION + 1
if COMPRESSION_METHOD == 2:
    DECODER_INPUT_CHANNEL = FEATURE_PYRAMID_G0_CHANNEL * pow(2, FP_DIMENSION) + FEATURE_PYRAMID_G1_CHANNEL + PE_CHANNEL * FP_DIMENSION + 1
if COMPRESSION_METHOD == 4 or COMPRESSION_METHOD == 5:
    DECODER_INPUT_CHANNEL = FEATURE_PYRAMID_G0_CHANNEL * pow(2, 2) + FEATURE_PYRAMID_G1_CHANNEL + PE_CHANNEL * FP_DIMENSION + 1

print("DECODER_INPUT_CHANNEL:", DECODER_INPUT_CHANNEL)
CROP_SIZE = pow(2, CROP_MIP_LEVEL)
MLP_DTYPE = bits2dtype_torch(MLP_NUM_DTYPE, "float")

SAVE_NAME = f"{PROJECT_NAME}_{DEVICE}_{BASENAME}_{TF_NO_MIP}_{COMPRESSION_METHOD}_{NUM_EPOCH}_{FEATURE_PYRAMID_SIZE_RATE}_{FEATURE_PYRAMID_G0_CHANNEL}_{FP_G0_BIT}_{FEATURE_PYRAMID_G1_CHANNEL}_{FP_G1_BIT}"

PRINTLOG_PATH = make_filename_by_seq("./printlog", f"{SAVE_NAME}.txt")
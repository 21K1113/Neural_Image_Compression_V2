import numpy as np
import cv2
import torch
import math
import glob
import os
import re
import random

from sympy.physics.units.systems.mksa import dimsys_MKSA
from torch.cuda import device


def judge_torf(arg, error_massage=""):
    value = arg.split("=")[1].lower()
    if value in ["true", "1"]:
        return True
    elif value in ["false", "0"]:
        return False
    else:
        raise ValueError(f"{error_massage} must be a boolean (True/False or 1/0)")


def judge_value(arg, dtype, error_massage=""):
    if dtype == "int":
        return int(arg.split("=")[1])
    if dtype == "float":
        return float(arg.split("=")[1])
    if dtype == "bool":
        return judge_torf(arg, error_massage)
    if dtype == "str":
        return f'"{arg.split("=")[1]}"'


def print_(arc_str, path):
    print(arc_str)
    with open(path, 'a') as f:
        print(arc_str, file=f)


def make_filename_by_seq(dirname, filename, seq_digit=3):
    if not os.path.exists(dirname):
        # ディレクトリが存在しない場合、ディレクトリを作成する
        os.makedirs(dirname)

    filename_without_ext, ext = os.path.splitext(filename)

    pattern = f"{filename_without_ext}_([0-9]*){ext}"
    prog = re.compile(pattern)

    files = glob.glob(
        os.path.join(dirname, f"{filename_without_ext}_[0-9]*{ext}")
    )

    max_seq = -1
    for f in files:
        m = prog.match(os.path.basename(f))
        if m:
            max_seq = max(max_seq, int(m.group(1)))

    new_filename = f"{dirname}/{filename_without_ext}_{max_seq + 1:0{seq_digit}}{ext}"

    return new_filename


# ファイルパスを取得しndarrayを返す
# https://qiita.com/teruto725/items/c8842bc7211a80e0e887
def readClip(filepass):
    cap = cv2.VideoCapture(filepass)
    print(cap.isOpened())  # 読み込めたかどうか
    ret, frame = cap.read()
    array = np.reshape(frame, (1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))
    while True:
        ret, frame = cap.read()  # 1フレーム読み込み
        if ret == False:  # フレームが読み込めなかったら出る
            break
        frame = np.reshape(frame,
                           (1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3))
        array = np.append(array, frame, axis=0)
    cap.release()
    return array


# numpyから動画に変換
# https://yusei-roadstar.hatenablog.com/entry/2019/11/29/174448
def timelaps(movie, saved_name, all_frame=64, width=64, height=64, frame_rate=32):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(saved_name, fourcc, frame_rate, (width, height))

    for i in range(all_frame):
        img = movie[i]
        video.write(img)

    video.release()
    print("動画変換完了")


def save_result_to_csv(result, filename):
    size = result.shape[0]# resultLUTのサイズを取得
    st = ''

    # resultLUTの内容をst形式に変換
    for diag_angle in range(size):
        for angle in range(size):
            for refangle in range(size):
                # 各colorの値をst形式に変換して追加
                for c in range(3):
                    st += str(result[diag_angle, angle, refangle, c].item())
                    st += ","
            st += "\n"  # refangleのループが終わったら改行を追加

    # CSVファイルに書き込む
    with open(filename, mode='w') as file:
        file.write(st)


# PSNRを計算する関数の定義
def calculate_psnr(original, reconstructed, num_bits=8):
    # print(original.shape)
    # print(reconstructed.shape)
    # print(torch.isnan(original).any())
    # print(torch.isinf(original).any())
    # print(torch.isnan(reconstructed).any())
    # print(torch.isinf(reconstructed).any())

    if isinstance(original, np.ndarray):
        mse = np.mean((original - reconstructed) ** 2)
    elif isinstance(original, torch.Tensor):
        mse = torch.mean((original - reconstructed) ** 2)

    if mse == 0:  # MSEが0の場合はPSNRは無限大
        return float('inf')
    max_pixel_value = pow(2, num_bits)
    if isinstance(original, np.ndarray):
        psnr = 10 * np.log10(max_pixel_value * max_pixel_value / mse)
    elif isinstance(original, torch.Tensor):
        psnr = 10 * torch.log10(max_pixel_value * max_pixel_value / mse)
    return psnr


# 元の動画と再構成された動画の各フレームに対してPSNRを計算し、平均PSNRを求める
def calculate_average_psnr(original_video, reconstructed_video):
    num_frames = original_video.shape[0]
    total_psnr = 0.0

    for i in range(num_frames):
        original_frame = original_video[i]
        reconstructed_frame = reconstructed_video[i]
        psnr = calculate_psnr(original_frame, reconstructed_frame)
        total_psnr += psnr

    average_psnr = total_psnr / num_frames
    return average_psnr


# tensorの最大値、最小値、平均、分散、NaNがあるか、Infがあるかを出力
def safe_statistics(tensor, printlog_path):
    # NaNとInfを除外するために、マスクを作成
    valid_mask = torch.isfinite(tensor)

    # 有効な値のみを取得
    valid_tensor = tensor[valid_mask]

    if valid_tensor.numel() == 0:
        print_("No valid numbers in the tensor.", printlog_path)
    else:
        # サイズ
        print_(f"Shape: {tensor.shape}", printlog_path)
        # 最大値
        max_val = torch.max(valid_tensor)
        print_(f"Max: {max_val.item()}", printlog_path)

        # 最小値
        min_val = torch.min(valid_tensor)
        print_(f"Min: {min_val.item()}", printlog_path)

        # 平均
        mean_val = torch.mean(valid_tensor)
        print_(f"Mean: {mean_val.item()}", printlog_path)

        # 分散
        var_val = torch.var(valid_tensor)
        print_(f"Variance: {var_val.item()}", printlog_path)

    # NaNの有無
    has_nan = torch.isnan(tensor).any()
    print_(f"Contains NaN: {has_nan.item()}", printlog_path)

    # Infの有無
    has_inf = torch.isinf(tensor).any()
    print_(f"Contains Inf: {has_inf.item()}", printlog_path)


"""
def positional_encoding(x, y, num_channels, device, dtype):  # (x, y) in (0, L/2)
    assert x.shape == y.shape, "x and y must have the same shape"
    pe = torch.zeros((x.shape[0], num_channels * 2), device=device, dtype=dtype)
    div_term = torch.exp(torch.arange(0, num_channels, 2, device=device, dtype=dtype) * -(math.log(10000.0) / num_channels))

    pe[:, 0:num_channels:2] = torch.sin(x.unsqueeze(-1) * div_term)
    pe[:, 1:num_channels:2] = torch.cos(x.unsqueeze(-1) * div_term)
    pe[:, num_channels::2] = torch.sin(y.unsqueeze(-1) * div_term)
    pe[:, num_channels+1::2] = torch.cos(y.unsqueeze(-1) * div_term)
    return pe.T
"""


def positional_encoding(coord, num_channels, device, dtype):  # (x, y) in (0, L/2)
    # assert x.shape == y.shape, "x and y must have the same shape"
    length = len(coord)
    pe = torch.zeros((coord[0].shape[0], num_channels * length), device=device, dtype=dtype)
    div_term = torch.exp(torch.arange(0, num_channels, 2, device=device, dtype=dtype) * -(math.log(10000.0) / num_channels))

    for i in range(length):
        pe[:, num_channels * i:num_channels * (i + 1):2] = torch.sin(coord[i].unsqueeze(-1) * div_term)
        pe[:, num_channels * i + 1:num_channels * (i + 1):2] = torch.cos(coord[i].unsqueeze(-1) * div_term)

    return pe.T


def triangular_positional_encoding(coord, num_channels, device, dtype):
    dimension = len(coord)
    octaves = num_channels // 2
    pe = torch.zeros((num_channels * dimension, coord[0].shape[0]), device=device, dtype=dtype)

    for octave in range(octaves):
        div = 2 ** octave
        for i, offset in enumerate((0.5, 0.0)):
            if octave == 0 and i == 0:
                continue
            pe[num_channels - (octave * 2 + i + 1):dimension * num_channels:num_channels, :] = tri(coord / div, offset=offset)

    return pe


def tri(x, offset=0.5):
    return 2 * torch.abs((x - offset) % 2 - 1) - 1


def triangular_positional_encoding_1d(device, dtype, sequence_length=8, octaves=3, include_constant=True):
    encodings = []
    x = torch.arange(0, sequence_length, step=1, device=device, dtype=dtype)
    for octave in range(octaves):
        div = 2 ** octave
        for i, offset in enumerate((0.0, 0.5)):
            if octave == 0 and i == 1:
                continue
            encoding = tri(x / div, offset=offset)
            encodings.append(encoding)
    if include_constant:
        encodings.append(torch.zeros(sequence_length, dtype=dtype, device=device))
    encodings = torch.stack(encodings)
    return encodings


def triangular_positional_encoding_2d(coordinates, h, w, device, dtype, sequence_length=8, octaves=3, stride=1, include_constant=True):
    encodings_1d = triangular_positional_encoding_1d(device, dtype, sequence_length, octaves, include_constant)
    b = coordinates.shape[0]
    # Convert coordinates for 2D positional encoding
    full_x, full_y = convert_coordinate_start(coordinates, h, w, device, dtype, stride)

    # Gather encodings based on x and y coordinates
    def get_encoding(full_coord):
        batch, seq_len = full_coord.shape
        d1, d2 = encodings_1d.shape
        encodings_expanded = encodings_1d.unsqueeze(0).expand(batch, d1, d2)
        full_coord_expanded = (full_coord % sequence_length).unsqueeze(1).expand(batch, d1, seq_len)
        return torch.gather(encodings_expanded, 2, full_coord_expanded)

    encoding_x = get_encoding(full_x).view(b, -1, h, w)
    encoding_y = get_encoding(full_y).view(b, -1, h, w)

    return torch.cat([encoding_x, encoding_y], dim=1)


def convert_coordinate_start(coordinate_start, h, w, device, dtype, stride=1, flatten_sequence=True):
    x_offset, y_offset = torch.arange(0, w*stride, step=stride, device=device), torch.arange(0, h*stride, step=stride, device=device)
    xx, yy = torch.meshgrid(x_offset, y_offset, indexing='ij')
    xx = xx.view(h, w, 1)
    yy = yy.view(h, w, 1)

    b = coordinate_start.shape[0]
    x_start, y_start = torch.split(coordinate_start, 1, dim=-1)
    # view as b x seq_len x 1
    x_start = x_start.view(b, 1, 1, 1)
    y_start = y_start.view(b, 1, 1, 1)

    full_x = x_start + xx
    full_y = y_start + yy
    if flatten_sequence:
        full_x = full_x.view(b, -1)
        full_y = full_y.view(b, -1)

    return full_x, full_y


def positional_encoding_3d(x, y, z, num_channels, device, dtype):  # (x, y) in (0, L/2)
    pe = torch.zeros((x.shape[0], num_channels * 3), device=device, dtype=dtype)
    div_term = torch.exp(torch.arange(0, num_channels, 2, device=device, dtype=dtype) * -(math.log(10000.0) / num_channels))

    pe[:, 0:num_channels:2] = torch.sin(x.unsqueeze(-1) * div_term)
    pe[:, 1:num_channels:2] = torch.cos(x.unsqueeze(-1) * div_term)
    pe[:, num_channels:num_channels * 2:2] = torch.sin(y.unsqueeze(-1) * div_term)
    pe[:, num_channels + 1:num_channels * 2:2] = torch.cos(y.unsqueeze(-1) * div_term)
    pe[:, num_channels * 2:num_channels * 3:2] = torch.sin(z.unsqueeze(-1) * div_term)
    pe[:, num_channels * 2 + 1:num_channels * 3:2] = torch.cos(z.unsqueeze(-1) * div_term)

    return pe.T


def bits2dtype_torch(num_bits, dtype="float"):
    if num_bits <= 8:
        return torch.uint8
    elif num_bits == 16 and dtype == "int":
        return torch.int16
    elif num_bits == 16 and dtype == "uint":
        return torch.uint16
    elif num_bits == 16 and dtype == "float":
        return torch.float16
    elif num_bits == 32:
        return torch.float32
    elif num_bits == 64:
        return torch.float64


def bits2dtype_np(num_bits, dtype="float"):
    if num_bits <= 8:
        return np.uint8
    elif num_bits == 16 and dtype == "int":
        return np.int16
    elif num_bits == 16 and dtype == "uint":
        return np.uint16
    elif num_bits == 16 and dtype == "float":
        return np.float16
    elif num_bits == 32:
        return np.float32
    elif num_bits == 64:
        return np.float64

def dtype_from_ext(ext):
    if ext.lower() in ['npy', 'npz']:
        return "ndarray"
    elif ext.lower() in ['avi', 'mp4']:
        return "movie"
    elif ext.lower() in ['png', 'jpg', 'jpeg']:
        return "image"


def diff_fps(fp1, fp2, printlog_path):
    for i in range(len(fp1)):
        diff = fp1[i] - fp2[i]
        print_("max" + str(torch.max(diff)), printlog_path)
        print_("min" + str(torch.min(diff)), printlog_path)
        print_("mean" + str(torch.mean(fp1[i])), printlog_path)
        print_("abs mean" + str(torch.mean(torch.abs(fp1[i]))), printlog_path)


def set_seed(seed):
    random.seed(seed)  # Pythonのrandomモジュールのシード
    np.random.seed(seed)  # NumPyのシード
    torch.manual_seed(seed)  # PyTorchのシード（CPU用）
    torch.cuda.manual_seed(seed)  # PyTorchのシード（GPU用）
    torch.cuda.manual_seed_all(seed)  # 複数GPU用
    torch.backends.cudnn.deterministic = True  # 再現性のための設定
    torch.backends.cudnn.benchmark = False  # パフォーマンスを犠牲にしても再現性を優先

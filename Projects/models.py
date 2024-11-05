import torch
import numpy as np

# 正規化数値からビット数値に変換
def scale_to_bit(tensor, bit=8):
    scale = pow(2, bit) - 1
    return tensor * scale


# ビット数値を正規化数値に変換
def normalize_from_bit(tensor, bit=8):
    scale = pow(2, bit) - 1
    return tensor / scale


# 量子化関数
def quantize_torch(tensor, num_bits):  # 入力：0~1 出力：0~1
    rounded_tensor = torch.floor(tensor * (pow(2, num_bits)-1) + 0.5)
    return rounded_tensor / (pow(2, num_bits)-1)


# 量子化関数
def quantize_np(ndy, num_bits):  # 入力：0~1 出力：0~1
    rounded_ndy = np.floor(ndy * (pow(2, num_bits)-1) + 0.5)
    return rounded_ndy / (pow(2, num_bits)-1)


# 量子化関数
def quantize(array, bit):  # 入力：0~1 出力：0~1
    if isinstance(array, np.ndarray):
        rounded_ndy = np.floor(array * (pow(2, bit) - 1) + 0.5)
        return rounded_ndy / (pow(2, bit) - 1)
    elif isinstance(array, torch.Tensor):
        rounded_tensor = torch.floor(array * (pow(2, bit) - 1) + 0.5)
        return rounded_tensor / (pow(2, bit) - 1)


# 量子化関数
def quantize_to_bit(array, num_bits=8):  # 入力：0~1 出力：0~2^bits
    return scale_to_bit(quantize(array, num_bits), num_bits)


# 量子化関数
def quantize_from_bit_to_bit(array, bit):  # 入力：0~2^bits 出力：0~2^bits
    return scale_to_bit(quantize(normalize_from_bit(array, bit), bit), bit)


def quantize_clamp(tensor, num_bits=8):
    q_min = -(pow(2, num_bits)-1)/pow(2, num_bits+1)
    q_max = 1/2
    return torch.clamp(tensor, min=q_min, max=q_max)


# fp用の量子化関数
def quantize4fp(tensor, num_bits):
    rounded_tensor = torch.floor(tensor * (pow(2, num_bits)-1) + 0.5)
    return rounded_tensor / (pow(2, num_bits)-1)


# 自然数に変換し、uint8にする
def save4fp(tensor, num_bits, dtype):
    rounded_tensor = torch.floor(tensor * (pow(2, num_bits)-1) + 0.5)
    positive_tensor = rounded_tensor + pow(2, num_bits - 1) - 1
    return positive_tensor.to(dtype)


# uint8のやつを元の配列に戻す
def load4fp(tensor, num_bits, dtype):
    float_tensor = tensor.to(dtype)
    zero_center_tensor = float_tensor - pow(2, num_bits - 1) + 1
    return zero_center_tensor / (pow(2, num_bits)-1)


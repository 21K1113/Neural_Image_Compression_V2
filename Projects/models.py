import torch
import numpy as np

# 正規化数値からビット数値に変換
def scale_to_bit(tensor, bit=8):
    scale = pow(2, bit) - 1
    return tensor * scale


# ビット数値を正規化数値に変換
def normalize_from_bit(tensor, bit):
    scale = pow(2, bit) - 1
    return tensor / scale


# 量子化関数
def quantize_from_norm_to_bit(array, bit):  # 入力：0~1 出力：0~2^bits
    scale = pow(2, bit) - 1
    if isinstance(array, np.ndarray):
        return np.floor(array * scale + 0.5)
    elif isinstance(array, torch.Tensor):
        return torch.floor(array * scale + 0.5)


# 量子化関数
def quantize_from_norm_to_norm(array, bit):  # 入力：0~1 出力：0~1
    return normalize_from_bit(quantize_from_norm_to_bit(array, bit), bit)


# 量子化関数
def quantize_from_bit_to_bit(array, bit):  # 入力：0~2^bits 出力：0~2^bits
    return quantize_from_norm_to_bit(normalize_from_bit(array, bit), bit)


# 量子化関数
def quantize_from_bit_to_norm(array, bit):  # 入力：0~2^bits 出力：0~1
    return normalize_from_bit(quantize_from_bit_to_bit(array, bit), bit)


def quantize_clamp(tensor, num_bits=8):
    q_min = -(pow(2, num_bits) - 1) / pow(2, num_bits + 1)
    q_max = 1 / 2
    return torch.clamp(tensor, min=q_min, max=q_max)


# fp用の量子化関数
def quantize4fp(tensor, num_bits):
    rounded_tensor = torch.floor(tensor * pow(2, num_bits) + 0.5)
    clamped_tensor = torch.clamp(rounded_tensor, min=-pow(2, num_bits-1)+1, max=pow(2, num_bits-1))
    return clamped_tensor / pow(2, num_bits)


# 自然数に変換し、uint8にする
def save4fp(tensor, num_bits, dtype):
    rounded_tensor = torch.floor(tensor * pow(2, num_bits) + 0.5)
    positive_tensor = rounded_tensor + pow(2, num_bits - 1) - 1
    return positive_tensor.to(dtype)


# uint8のやつを元の配列に戻す
def load4fp(tensor, num_bits, mlt_dtype):
    float_tensor = tensor.to(mlt_dtype)
    zero_center_tensor = float_tensor - pow(2, num_bits - 1) + 1
    return zero_center_tensor / pow(2, num_bits)


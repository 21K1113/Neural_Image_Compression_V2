import torch
from collections import defaultdict
from models import quantize4fp, save4fp, load4fp
from utils import positional_encoding, triangular_positional_encoding


# 2のn乗を返す logでよくね
def return_2_power(base_size):
    count = 0
    x = base_size
    while x != 1:
        x = x // 2
        count += 1
    return count


# 特徴ピラミッドの最大レベルを返す
def return_pyramid_levels(base_size):
    count = return_2_power(base_size)
    return (count + 1) // 2


# 特徴ピラミッドのレベルとミップレベルの関係の辞書を返す
def create_pyramid_mip_levels(image_size, base_size):
    count = return_2_power(image_size)
    feature_pyramid_dict = defaultdict(int)
    levels = return_pyramid_levels(base_size)
    for i in range(count + 1):
        feature_pyramid_dict[i] = (i // 2) - 1
        if feature_pyramid_dict[i] < 0:
            feature_pyramid_dict[i] = 0
        elif feature_pyramid_dict[i] >= levels:
            feature_pyramid_dict[i] = levels - 1
    return feature_pyramid_dict


def create_pyramid(base_size, channels, num_bits, device, dtype, no_mip=False):
    """
    ピラミッド構造の配列を作成する関数
    :param base_size: 最下層の配列のサイズ (base_size, base_size)
    :param channels: 配列のチャンネル数
    :param levels: ピラミッドのレベル数
    :return: ピラミッド構造の配列（リスト形式）
    """
    levels = return_pyramid_levels(base_size)
    if no_mip:
        levels = 1
    q_min = -(pow(2, num_bits) - 1) / pow(2, num_bits + 1)
    q_max = 1 / 2
    pyramid = []
    for i in range(levels * 2):
        size = base_size // (2 ** i)
        # array = torch.randn(channels, size + 1, size + 1, device=device, requires_grad=True)
        array = ((q_max - q_min) * torch.rand(channels, size + 1, size + 1, device=device, dtype=dtype) + q_min).requires_grad_(True)
        pyramid.append(array)
    return pyramid, levels


def create_pyramid_3d(base_size, channels, num_bits, device, dtype, no_mip=False):
    """
    ピラミッド構造の配列を作成する関数
    :param base_size: 最下層の配列のサイズ (base_size, base_size)
    :param channels: 配列のチャンネル数
    :param levels: ピラミッドのレベル数
    :return: ピラミッド構造の配列（リスト形式）
    """
    levels = return_pyramid_levels(base_size)
    if no_mip:
        levels = 1
    q_min = -(pow(2, num_bits) - 1) / pow(2, num_bits + 1)
    q_max = 1 / 2
    pyramid = []
    for i in range(levels * 2):
        size = base_size // (2 ** i)
        # array = torch.randn(channels, size + 1, size + 1, device=device, requires_grad=True)
        array = ((q_max - q_min) * torch.rand(channels, size + 1, size + 1, size + 1, device=device, dtype=dtype) + q_min).requires_grad_(True)
        pyramid.append(array)
    return pyramid, levels


def create_g(fp, fl, j, x_indices, y_indices):
    g_0 = fp[fl * 2 + j][:, y_indices, x_indices]
    g_1 = fp[fl * 2 + j][:, y_indices + 1, x_indices]
    g_2 = fp[fl * 2 + j][:, y_indices, x_indices + 1]
    g_3 = fp[fl * 2 + j][:, y_indices + 1, x_indices + 1]
    return g_0, g_1, g_2, g_3


def create_g_3d(fp, fl, j, x_indices, y_indices, z_indices):
    # torch.set_printoptions(edgeitems=1000)
    # print(z_indices)
    # print(fp[fl * 2 + j].shape[1])
    # assert torch.all(z_indices + 1 < fp[fl * 2 + j].shape[1])
    # assert torch.all(y_indices + 1 < fp[fl * 2 + j].shape[2])
    # assert torch.all(x_indices + 1 < fp[fl * 2 + j].shape[3])
    g_0 = fp[fl * 2 + j][:, z_indices, y_indices, x_indices]
    g_1 = fp[fl * 2 + j][:, z_indices + 1, y_indices, x_indices]
    g_2 = fp[fl * 2 + j][:, z_indices, y_indices + 1, x_indices]
    g_3 = fp[fl * 2 + j][:, z_indices + 1, y_indices + 1, x_indices]
    g_4 = fp[fl * 2 + j][:, z_indices, y_indices, x_indices + 1]
    g_5 = fp[fl * 2 + j][:, z_indices + 1, y_indices, x_indices + 1]
    g_6 = fp[fl * 2 + j][:, z_indices, y_indices + 1, x_indices + 1]
    g_7 = fp[fl * 2 + j][:, z_indices + 1, y_indices + 1, x_indices + 1]
    return g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7


def create_g_3d_v2(fp, fl, j, x_indices, y_indices, z_indices):
    g_0 = fp[fl * 2 + j][:, z_indices, y_indices, x_indices]
    g_1 = fp[fl * 2 + j][:, z_indices + 1, y_indices + 1, x_indices]
    g_2 = fp[fl * 2 + j][:, z_indices + 1, y_indices, x_indices + 1]
    g_3 = fp[fl * 2 + j][:, z_indices, y_indices + 1, x_indices + 1]
    return g_0, g_1, g_2, g_3


def create_g0_g1(fp, fl, x, y, step_number, x_range, y_range, pe_channels, device, dtype, use_tri_pe=True):
    x_g0_tensor = (x_range + x) * step_number
    y_g0_tensor = (y_range + y) * step_number
    x_g0_index = torch.floor(x_g0_tensor).to(torch.int)
    y_g0_index = torch.floor(y_g0_tensor).to(torch.int)
    x_g1_tensor = x_g0_tensor / 2
    y_g1_tensor = y_g0_tensor / 2
    x_g1_index = torch.floor(x_g1_tensor).to(torch.int)
    y_g1_index = torch.floor(y_g1_tensor).to(torch.int)
    x_g0_grid, y_g0_grid = torch.meshgrid(x_g0_index, y_g0_index, indexing='ij')
    x_g1_grid, y_g1_grid = torch.meshgrid(x_g1_index, y_g1_index, indexing='ij')
    x_pe_grid, y_pe_grid = torch.meshgrid(x_g1_tensor, y_g1_tensor, indexing='ij')
    x_g0_indices, y_g0_indices = x_g0_grid.reshape(-1), y_g0_grid.reshape(-1)
    x_g1_indices, y_g1_indices = x_g1_grid.reshape(-1), y_g1_grid.reshape(-1)
    x_pe_indices, y_pe_indices = x_pe_grid.reshape(-1), y_pe_grid.reshape(-1)
    g0_0, g0_1, g0_2, g0_3 = create_g(fp, fl, 0, x_g0_indices, y_g0_indices)
    g1_0, g1_1, g1_2, g1_3 = create_g(fp, fl, 1, x_g1_indices, y_g1_indices)
    if use_tri_pe:
        pe = triangular_positional_encoding(torch.stack([x_pe_indices, y_pe_indices]), pe_channels, device, dtype)
    else:
        pe = positional_encoding((x_pe_indices, y_pe_indices), pe_channels, device, dtype)
    if int(1 // (step_number / 2)) != 1:
        x_g1_k = x_g1_tensor - x_g1_index.to(dtype)
        y_g1_k = y_g1_tensor - y_g1_index.to(dtype)
        x_g1_k_grid, y_g1_k_grid = torch.meshgrid(x_g1_k, y_g1_k, indexing='ij')
        x_g1_k_indices, y_g1_k_indices = x_g1_k_grid.reshape(-1), y_g1_k_grid.reshape(-1)
        g1_0 = g1_0 * (1 - x_g1_k_indices) * (1 - y_g1_k_indices)
        g1_1 = g1_1 * (1 - x_g1_k_indices) * y_g1_k_indices
        g1_2 = g1_2 * x_g1_k_indices * (1 - y_g1_k_indices)
        g1_3 = g1_3 * x_g1_k_indices * y_g1_k_indices
    return g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, pe


def create_g0_g1_3d(fp, fl, x, y, z, step_number, x_range, y_range, z_range, pe_channels, device, dtype):
    x_g0_tensor = (x_range + x) * step_number
    y_g0_tensor = (y_range + y) * step_number
    z_g0_tensor = (z_range + z) * step_number
    x_g0_index = torch.floor(x_g0_tensor).to(torch.int)
    y_g0_index = torch.floor(y_g0_tensor).to(torch.int)
    z_g0_index = torch.floor(z_g0_tensor).to(torch.int)
    x_g1_tensor = x_g0_tensor / 2
    y_g1_tensor = y_g0_tensor / 2
    z_g1_tensor = z_g0_tensor / 2
    x_g1_index = torch.floor(x_g1_tensor).to(torch.int)
    y_g1_index = torch.floor(y_g1_tensor).to(torch.int)
    z_g1_index = torch.floor(z_g1_tensor).to(torch.int)
    x_g0_grid, y_g0_grid, z_g0_grid = torch.meshgrid(x_g0_index, y_g0_index, z_g0_index, indexing='ij')
    x_g1_grid, y_g1_grid, z_g1_grid = torch.meshgrid(x_g1_index, y_g1_index, z_g1_index, indexing='ij')
    x_pe_grid, y_pe_grid, z_pe_grid = torch.meshgrid(x_g1_tensor, y_g1_tensor, z_g1_tensor, indexing='ij')
    x_g0_indices, y_g0_indices, z_g0_indices = x_g0_grid.reshape(-1), y_g0_grid.reshape(-1), z_g0_grid.reshape(-1)
    x_g1_indices, y_g1_indices, z_g1_indices = x_g1_grid.reshape(-1), y_g1_grid.reshape(-1), z_g1_grid.reshape(-1)
    x_pe_indices, y_pe_indices, z_pe_indices = x_pe_grid.reshape(-1), y_pe_grid.reshape(-1), z_pe_grid.reshape(-1)
    g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7 = create_g_3d(fp, fl, 0, x_g0_indices, y_g0_indices, z_g0_indices)
    g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7 = create_g_3d(fp, fl, 1, x_g1_indices, y_g1_indices, z_g1_indices)
    pe = triangular_positional_encoding(torch.stack([x_pe_indices, y_pe_indices, z_pe_indices]), pe_channels, device, dtype)
    if int(1 // (step_number / 2)) != 1:
        x_g1_k = x_g1_tensor - x_g1_index.to(dtype)
        y_g1_k = y_g1_tensor - y_g1_index.to(dtype)
        z_g1_k = z_g1_tensor - z_g1_index.to(dtype)
        x_g1_k_grid, y_g1_k_grid, z_g1_k_grid = torch.meshgrid(x_g1_k, y_g1_k, z_g1_k, indexing='ij')
        x_g1_k_indices, y_g1_k_indices, z_g1_k_indices = (x_g1_k_grid.reshape(-1), y_g1_k_grid.reshape(-1), z_g1_k_grid.reshape(-1))
        g1_0 = g1_0 * (1 - x_g1_k_indices) * (1 - y_g1_k_indices) * (1 - z_g1_k_indices)
        g1_1 = g1_1 * (1 - x_g1_k_indices) * (1 - y_g1_k_indices) * z_g1_k_indices
        g1_2 = g1_2 * (1 - x_g1_k_indices) * y_g1_k_indices * (1 - z_g1_k_indices)
        g1_3 = g1_3 * x_g1_k_indices * (1 - y_g1_k_indices) * (1 - z_g1_k_indices)
        g1_4 = g1_4 * x_g1_k_indices * y_g1_k_indices * (1 - z_g1_k_indices)
        g1_5 = g1_5 * x_g1_k_indices * (1 - y_g1_k_indices) * z_g1_k_indices
        g1_6 = g1_6 * (1 - x_g1_k_indices) * y_g1_k_indices * z_g1_k_indices
        g1_7 = g1_7 * x_g1_k_indices * y_g1_k_indices * z_g1_k_indices
    return g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7, g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7, pe


def create_g0_g1_3d_v2(fp, fl, x, y, z, step_number, x_range, y_range, z_range, pe_channels, device, dtype):
    x_g0_tensor = (x_range + x) * step_number
    y_g0_tensor = (y_range + y) * step_number
    z_g0_tensor = (z_range + z) * step_number
    x_g0_index = torch.floor(x_g0_tensor).to(torch.int)
    y_g0_index = torch.floor(y_g0_tensor).to(torch.int)
    z_g0_index = torch.floor(z_g0_tensor).to(torch.int)
    x_g1_tensor = x_g0_tensor / 2
    y_g1_tensor = y_g0_tensor / 2
    z_g1_tensor = z_g0_tensor / 2
    x_g1_index = torch.floor(x_g1_tensor).to(torch.int)
    y_g1_index = torch.floor(y_g1_tensor).to(torch.int)
    z_g1_index = torch.floor(z_g1_tensor).to(torch.int)
    x_g0_grid, y_g0_grid, z_g0_grid = torch.meshgrid(x_g0_index, y_g0_index, z_g0_index, indexing='ij')
    x_g1_grid, y_g1_grid, z_g1_grid = torch.meshgrid(x_g1_index, y_g1_index, z_g1_index, indexing='ij')
    x_pe_grid, y_pe_grid, z_pe_grid = torch.meshgrid(x_g1_tensor, y_g1_tensor, z_g1_tensor, indexing='ij')
    x_g0_indices, y_g0_indices, z_g0_indices = x_g0_grid.reshape(-1), y_g0_grid.reshape(-1), z_g0_grid.reshape(-1)
    x_g1_indices, y_g1_indices, z_g1_indices = x_g1_grid.reshape(-1), y_g1_grid.reshape(-1), z_g1_grid.reshape(-1)
    x_pe_indices, y_pe_indices, z_pe_indices = x_pe_grid.reshape(-1), y_pe_grid.reshape(-1), z_pe_grid.reshape(-1)
    g0_0, g0_1, g0_2, g0_3 = create_g_3d_v2(fp, fl, 0, x_g0_indices, y_g0_indices, z_g0_indices)
    g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7 = create_g_3d(fp, fl, 1, x_g1_indices, y_g1_indices, z_g1_indices)
    pe = positional_encoding((x_pe_indices, y_pe_indices, z_pe_indices), pe_channels, device, dtype)
    if int(1 // (step_number / 2)) != 1:
        x_g1_k = x_g1_tensor - x_g1_index.to(torch.float)
        y_g1_k = y_g1_tensor - y_g1_index.to(torch.float)
        z_g1_k = z_g1_tensor - z_g1_index.to(torch.float)
        x_g1_k_grid, y_g1_k_grid, z_g1_k_grid = torch.meshgrid(x_g1_k, y_g1_k, z_g1_k, indexing='ij')
        x_g1_k_indices, y_g1_k_indices, z_g1_k_indices = (x_g1_k_grid.reshape(-1), y_g1_k_grid.reshape(-1), z_g1_k_grid.reshape(-1))
        g1_0 = g1_0 * (1 - x_g1_k_indices) * (1 - y_g1_k_indices) * (1 - z_g1_k_indices)
        g1_1 = g1_1 * (1 - x_g1_k_indices) * (1 - y_g1_k_indices) * z_g1_k_indices
        g1_2 = g1_2 * (1 - x_g1_k_indices) * y_g1_k_indices * (1 - z_g1_k_indices)
        g1_3 = g1_3 * x_g1_k_indices * (1 - y_g1_k_indices) * (1 - z_g1_k_indices)
        g1_4 = g1_4 * x_g1_k_indices * y_g1_k_indices * (1 - z_g1_k_indices)
        g1_5 = g1_5 * x_g1_k_indices * (1 - y_g1_k_indices) * z_g1_k_indices
        g1_6 = g1_6 * (1 - x_g1_k_indices) * y_g1_k_indices * z_g1_k_indices
        g1_7 = g1_7 * x_g1_k_indices * y_g1_k_indices * z_g1_k_indices
    return g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7, pe


# fpをクランプする
def fp_quantize_clamp(fp, fl, num_bits):
    q_min = -(pow(2, num_bits) - 1) / pow(2, num_bits + 1)
    q_max = 1 / 2
    with torch.no_grad():
        fp[fl * 2].clamp_(q_min, q_max)
        fp[fl * 2 + 1].clamp_(q_min, q_max)


# fpを量子化する（クランプされていることが前提）
def fp_quantize(fp, fl, num_bits):
    with torch.no_grad():
        fp[fl * 2] = quantize4fp(fp[fl * 2], num_bits)
        fp[fl * 2 + 1] = quantize4fp(fp[fl * 2 + 1], num_bits)


def fp_all_quantize(fp, num_bits):
    quantized_fp = []
    for g in fp:
        quantized_g = quantize4fp(g, num_bits)
        quantized_fp.append(quantized_g)
    return quantized_fp


def fp_savable(fp, num_bits, dtype):
    compressed_fp = []
    for g in fp:
        compressed_g = save4fp(g, num_bits, dtype)
        compressed_fp.append(compressed_g)
    return compressed_fp


def fp_load(compressed_fp, num_bits, dtype):
    fp = []
    for compressed_g in compressed_fp:
        g = load4fp(compressed_g, num_bits, dtype)
        fp.append(g)
    return fp


def fp_freeze(fp):
    for g in fp:
        g.requires_grad = False

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


# 特徴ピラミッドの最大レベルを返す
def return_pyramid_levels_from_power(fp_power):
    return (fp_power + 1) // 2


# 特徴ピラミッドのレベルとミップレベルの関係の辞書を返す
def create_pyramid_mip_levels(image_size, fp_size_rate):
    image_size_power = return_2_power(image_size)
    fp_level = return_pyramid_levels_from_power(image_size_power - fp_size_rate)
    feature_pyramid_dict = defaultdict(int)
    for i in range(image_size_power + 1):
        feature_pyramid_dict[i] = min(max(0, (i - fp_size_rate) // 2), fp_level - 1)
    return feature_pyramid_dict


def create_pyramid(base_size, dim, g0_channel, g0_bit, g1_channel, g1_bit, device, dtype, no_mip=False):
    """
    ピラミッド構造の配列を作成する関数
    :param base_size: 最下層の配列のサイズ
    :param dim: 配列の次元
    :param g0_bit: G0の量子化ビット数
    :param g0_channel: G0のチャンネル数
    :param g1_bit: G1の量子化ビット数
    :param g1_channel: G1のチャンネル数
    :return: ピラミッド構造の配列（リスト形式）
    """
    fp_level = return_pyramid_levels(base_size)
    if no_mip:
        fp_level = 1
    pyramid = []
    for i in range(fp_level * 2):
        size = base_size // (2 ** i)
        if i % 2 == 0:
            channel = g0_channel
            bit = g0_bit
        else:  # i % 2 == 1
            channel = g1_channel
            bit = g1_bit
        if dim == 2:
            array = torch.rand(channel, size + 1, size + 1, device=device, dtype=dtype)
        elif dim == 3:
            array = torch.rand(channel, size + 1, size + 1, size + 1, device=device, dtype=dtype)
        q_min = -(pow(2, bit) - 1) / pow(2, bit + 1)
        q_max = 1 / 2
        grid = ((q_max - q_min) * array + q_min).requires_grad_(True)
        pyramid.append(grid)
    return pyramid, fp_level


def create_g(fp, fl, j, x_indices, y_indices):
    g_0 = fp[fl * 2 + j][:, y_indices, x_indices]
    g_1 = fp[fl * 2 + j][:, y_indices + 1, x_indices]
    g_2 = fp[fl * 2 + j][:, y_indices, x_indices + 1]
    g_3 = fp[fl * 2 + j][:, y_indices + 1, x_indices + 1]
    return g_0, g_1, g_2, g_3


def create_g_3d(fp, fl, j, x_indices, y_indices, z_indices):
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


def create_g1_k(g1, g1_k_grid):
    g1[0] = g1[0] * (1 - g1_k_grid[0]) * (1 - g1_k_grid[1])
    g1[1] = g1[1] * (1 - g1_k_grid[0]) * g1_k_grid[1]
    g1[2] = g1[2] * g1_k_grid[0] * (1 - g1_k_grid[1])
    g1[3] = g1[3] * g1_k_grid[0] * g1_k_grid[1]
    return g1


def create_g1_k_3d(g1, g1_k_grid):
    g1[0] = g1[0] * (1 - g1_k_grid[0]) * (1 - g1_k_grid[1]) * (1 - g1_k_grid[2])
    g1[1] = g1[1] * (1 - g1_k_grid[0]) * (1 - g1_k_grid[1]) * g1_k_grid[2]
    g1[2] = g1[2] * (1 - g1_k_grid[0]) * g1_k_grid[1] * (1 - g1_k_grid[2])
    g1[3] = g1[3] * g1_k_grid[0] * (1 - g1_k_grid[1]) * (1 - g1_k_grid[2])
    g1[4] = g1[4] * g1_k_grid[0] * g1_k_grid[1] * (1 - g1_k_grid[2])
    g1[5] = g1[5] * g1_k_grid[0] * (1 - g1_k_grid[1]) * g1_k_grid[2]
    g1[6] = g1[6] * (1 - g1_k_grid[0]) * g1_k_grid[1] * g1_k_grid[2]
    g1[7] = g1[7] * g1_k_grid[0] * g1_k_grid[1] * g1_k_grid[2]
    return g1


def create_meshgrid(any_indices):
    any_grid = torch.meshgrid(*any_indices, indexing='ij')
    return [grid.reshape(-1) for grid in any_grid]


def create_g0_g1(fp, fl, coord, step_number, pe_step_number, sample_ranges, pe_channels, method, device, dtype):
    range_add_coord = sample_ranges + coord  # (2, l) or (3, l)
    step_tensor = range_add_coord * step_number
    g0_indices = torch.floor(step_tensor).to(torch.int)
    g1_indices = (step_tensor // 2).to(torch.int)
    pe_indices = range_add_coord * pe_step_number
    g0_flat = create_meshgrid(g0_indices)
    g1_flat = create_meshgrid(g1_indices)
    pe_flat = create_meshgrid(pe_indices)
    if method == 1 or method == 2:
        g0 = create_g(fp, fl, 0, *g0_flat)  # tuple
        g1 = create_g(fp, fl, 1, *g1_flat)  # tuple
    elif method == 3:
        g0 = create_g_3d(fp, fl, 0, *g0_flat)  # tuple
        g1 = create_g_3d(fp, fl, 1, *g1_flat)  # tuple
    elif method == 4:
        g0 = create_g_3d_v2(fp, fl, 0, *g0_flat)  # tuple
        g1 = create_g_3d(fp, fl, 1, *g1_flat)  # tuple
    pe = triangular_positional_encoding(torch.stack(pe_flat), pe_channels, device, dtype)
    g1_k = (range_add_coord + 0.5) * step_number % 1
    g1_k_grid = create_meshgrid(g1_k)
    if method == 1 or method == 2:
        g1 = create_g1_k(list(g1), g1_k_grid)
    elif method == 3 or method == 4:
        g1 = create_g1_k_3d(list(g1), g1_k_grid)
    stacked_g1 = torch.stack(g1)
    sum_g1 = torch.sum(stacked_g1, dim=0)
    return *g0, sum_g1, pe


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


# fpをすべて量子化する  float -> float
def fp_all_quantize(fp, g0_bits, g1_bits, miss=False):
    quantized_fp = []
    for i, g in enumerate(fp):
        if i % 2 == 0:
            quantized_g = quantize4fp(g, g0_bits, miss)
        else:  # i % 2 == 1
            quantized_g = quantize4fp(g, g1_bits, miss)
        quantized_fp.append(quantized_g)
    return quantized_fp


# fpをすべて量子化する  float -> arg dtype
def fp_savable(fp, g0_bits, g0_dtype, g1_bits, g1_dtype):
    compressed_fp = []
    for i, g in enumerate(fp):
        if i % 2 == 0:
            compressed_g = save4fp(g, g0_bits, g0_dtype)
        else:  # i % 2 == 1
            compressed_g = save4fp(g, g1_bits, g1_dtype)
        compressed_fp.append(compressed_g)
    return compressed_fp


# fpをすべて量子化する  arg dtype -> float
def fp_load(compressed_fp, g0_bits, g1_bits, mlt_dtype):
    fp = []
    for i, compressed_g in enumerate(compressed_fp):
        if i % 2 == 0:
            g = load4fp(compressed_g, g0_bits, mlt_dtype)
        else:
            g = load4fp(compressed_g, g1_bits, mlt_dtype)
        fp.append(g)
    return fp


def fp_freeze(fp):
    for g in fp:
        g.requires_grad = False

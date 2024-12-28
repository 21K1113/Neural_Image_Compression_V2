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
            array = torch.rand(channel, size, size, device=device, dtype=dtype)
        elif dim == 3:
            array = torch.rand(channel, size, size, size, device=device, dtype=dtype)
        q_min = -(pow(2, bit) - 1) / pow(2, bit + 1)
        q_max = 1 / 2
        grid = ((q_max - q_min) * array + q_min).requires_grad_(True)
        pyramid.append(grid)
    return pyramid, fp_level


def create_g(fp, fl, j, x_indices, y_indices):
    g = fp[fl * 2 + j]
    height, width = g.shape[-2], g.shape[-1]

    x0 = torch.clamp(x_indices, 0, width - 1)
    x1 = torch.clamp(x_indices + 1, 0, width - 1)
    y0 = torch.clamp(y_indices, 0, height - 1)
    y1 = torch.clamp(y_indices + 1, 0, height - 1)

    g_0 = g[:, y0, x0]
    g_1 = g[:, y1, x0]
    g_2 = g[:, y0, x1]
    g_3 = g[:, y1, x1]
    return g_0, g_1, g_2, g_3


def create_g_3d(fp, fl, j, x_indices, y_indices, z_indices):
    g = fp[fl * 2 + j]
    depth, height, width = g.shape[-3], g.shape[-2], g.shape[-1]

    x0 = torch.clamp(x_indices, 0, width - 1)
    x1 = torch.clamp(x_indices + 1, 0, width - 1)
    y0 = torch.clamp(y_indices, 0, height - 1)
    y1 = torch.clamp(y_indices + 1, 0, height - 1)
    z0 = torch.clamp(z_indices, 0, depth - 1)
    z1 = torch.clamp(z_indices + 1, 0, depth - 1)

    g_0 = g[:, z0, y0, x0]
    g_1 = g[:, z1, y0, x0]
    g_2 = g[:, z0, y1, x0]
    g_3 = g[:, z1, y1, x0]
    g_4 = g[:, z0, y0, x1]
    g_5 = g[:, z1, y0, x1]
    g_6 = g[:, z0, y1, x1]
    g_7 = g[:, z1, y1, x1]
    return g_0, g_1, g_2, g_3, g_4, g_5, g_6, g_7


def create_g_3d_v2(fp, fl, j, x_indices, y_indices, z_indices):
    g = fp[fl * 2 + j]
    depth, height, width = g.shape[-3], g.shape[-2], g.shape[-1]

    # Clamp indices with a mask to handle boundary conditions
    x0 = torch.clamp(x_indices, 0, width - 1)
    x1 = torch.clamp(x_indices + 1, 0, width - 1)
    y0 = torch.clamp(y_indices, 0, height - 1)
    y1 = torch.clamp(y_indices + 1, 0, height - 1)
    z0 = torch.clamp(z_indices, 0, depth - 1)
    z1 = torch.clamp(z_indices + 1, 0, depth - 1)

    g_0 = g[:, z0, y0, x0]
    g_1 = g[:, z1, y1, x0]
    g_2 = g[:, z1, y0, x1]
    g_3 = g[:, z0, y1, x1]
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


def create_g0_g1_pe(fp, fl, coord, step_number, pe_step_number, sample_ranges, pe_channels, method, device, dtype):
    range_add_coord = sample_ranges + coord  # (2, l) or (3, l)
    step_tensor = (range_add_coord + 0.5) * step_number - 0.5
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
    return g0, g1, pe


def create_sum_g1(g1, sample_ranges, coord, step_number, method):
    g1_k = (sample_ranges + coord + 0.5) * step_number / 2 % 1
    g1_k_grid = create_meshgrid(g1_k)
    if method == 1 or method == 2:
        g1 = create_g1_k(list(g1), g1_k_grid)
    elif method == 3 or method == 4:
        g1 = create_g1_k_3d(list(g1), g1_k_grid)
    stacked_g1 = torch.stack(g1)
    sum_g1 = torch.sum(stacked_g1, dim=0)
    return sum_g1


# fpをクランプする
def fp_quantize_clamp(fp, fl, g0_bit, g1_bit):
    q0_min = -(pow(2, g0_bit) - 1) / pow(2, g0_bit + 1)
    q0_max = 1 / 2
    q1_min = -(pow(2, g1_bit) - 1) / pow(2, g1_bit + 1)
    q1_max = 1 / 2
    with torch.no_grad():
        fp[fl * 2].clamp_(q0_min, q0_max)
        fp[fl * 2 + 1].clamp_(q1_min, q1_max)


# fpを量子化する（クランプされていることが前提）
def fp_quantize(fp, fl, num_bits):
    with torch.no_grad():
        fp[fl * 2] = quantize4fp(fp[fl * 2], num_bits)
        fp[fl * 2 + 1] = quantize4fp(fp[fl * 2 + 1], num_bits)


# fpをすべて量子化する  float -> float
def fp_all_quantize(fp, g0_bits, g1_bits):
    quantized_fp = []
    for i, g in enumerate(fp):
        if i % 2 == 0:
            quantized_g = quantize4fp(g, g0_bits)
        else:  # i % 2 == 1
            quantized_g = quantize4fp(g, g1_bits)
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




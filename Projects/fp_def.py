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


def create_g0_g1_3d(fp, fl, x, y, z, step_number, pe_step_number, sample_range, pe_channels, device, dtype):
    x_g0_tensor = (sample_range + x) * step_number
    y_g0_tensor = (sample_range + y) * step_number
    z_g0_tensor = (sample_range + z) * step_number
    x_g0_index = torch.floor(x_g0_tensor).to(torch.int)
    y_g0_index = torch.floor(y_g0_tensor).to(torch.int)
    z_g0_index = torch.floor(z_g0_tensor).to(torch.int)
    x_g1_index = torch.floor(x_g0_tensor / 2).to(torch.int)
    y_g1_index = torch.floor(y_g0_tensor / 2).to(torch.int)
    z_g1_index = torch.floor(z_g0_tensor / 2).to(torch.int)
    x_pe_tensor = (sample_range + x) * pe_step_number
    y_pe_tensor = (sample_range + y) * pe_step_number
    z_pe_tensor = (sample_range + z) * pe_step_number
    x_g0_grid, y_g0_grid, z_g0_grid = torch.meshgrid(x_g0_index, y_g0_index, z_g0_index, indexing='ij')
    x_g1_grid, y_g1_grid, z_g1_grid = torch.meshgrid(x_g1_index, y_g1_index, z_g1_index, indexing='ij')
    x_pe_grid, y_pe_grid, z_pe_grid = torch.meshgrid(x_pe_tensor, y_pe_tensor, z_pe_tensor, indexing='ij')
    x_g0_indices, y_g0_indices, z_g0_indices = x_g0_grid.reshape(-1), y_g0_grid.reshape(-1), z_g0_grid.reshape(-1)
    x_g1_indices, y_g1_indices, z_g1_indices = x_g1_grid.reshape(-1), y_g1_grid.reshape(-1), z_g1_grid.reshape(-1)
    x_pe_indices, y_pe_indices, z_pe_indices = x_pe_grid.reshape(-1), y_pe_grid.reshape(-1), z_pe_grid.reshape(-1)
    g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7 = create_g_3d(fp, fl, 0, x_g0_indices, y_g0_indices, z_g0_indices)
    g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7 = create_g_3d(fp, fl, 1, x_g1_indices, y_g1_indices, z_g1_indices)
    pe = triangular_positional_encoding(torch.stack([x_pe_indices, y_pe_indices, z_pe_indices]), pe_channels, device, dtype)
    if int(1 // (step_number / 2)) != 1:
        x_g1_k = ((sample_range + x + 1/2) * step_number) % 1
        y_g1_k = ((sample_range + y + 1/2) * step_number) % 1
        z_g1_k = ((sample_range + z + 1/2) * step_number) % 1
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


def create_g0_g1_3d_test(fp, fl, coord, step_number, pe_step_number, sample_ranges, pe_channels, device, dtype):
    range_xyz = sample_ranges + coord
    step_tensor = range_xyz * step_number

    # g0とg1のインデックス計算
    g0_indices = torch.floor(step_tensor).to(torch.int)
    g1_indices = torch.floor(step_tensor / 2).to(torch.int)

    # Positional Encodingの計算
    pe_tensor = range_xyz * pe_step_number
    pe_indices = pe_tensor.reshape(3, -1)  # (3, 3*l) に変換

    g0_grid = torch.meshgrid(*g0_indices, indexing='ij')
    g1_grid = torch.meshgrid(*g1_indices, indexing='ij')
    pe_grid = torch.meshgrid(*pe_indices, indexing='ij')

    g0_flat = [grid.reshape(-1) for grid in g0_grid]
    g1_flat = [grid.reshape(-1) for grid in g1_grid]
    pe_flat = [grid.reshape(-1) for grid in pe_grid]

    # G0とG1の作成
    g0 = create_g_3d(fp, fl, 0, *g0_flat)
    g1 = create_g_3d(fp, fl, 1, *g1_flat)

    # Positional Encoding生成
    pe = triangular_positional_encoding(torch.stack(pe_flat), pe_channels, device, dtype)

    # インターポレーション処理（必要な場合のみ）
    if int(1 // (step_number / 2)) != 1:
        g1_k = (range_xyz + 0.5) * step_number % 1
        g1_k_grid = torch.meshgrid(*g1_k, indexing='ij')
        g1_k_grid = [grid.reshape(-1) for grid in g1_k_grid]
        g1 = list(g1)
        g1[0] = g1[0] * (1 - g1_k_grid[0]) * (1 - g1_k_grid[1]) * (1 - g1_k_grid[2])
        g1[1] = g1[1] * (1 - g1_k_grid[0]) * (1 - g1_k_grid[1]) * g1_k_grid[2]
        g1[2] = g1[2] * (1 - g1_k_grid[0]) * g1_k_grid[1] * (1 - g1_k_grid[2])
        g1[3] = g1[3] * g1_k_grid[0] * (1 - g1_k_grid[1]) * (1 - g1_k_grid[2])
        g1[4] = g1[4] * g1_k_grid[0] * g1_k_grid[1] * (1 - g1_k_grid[2])
        g1[5] = g1[5] * g1_k_grid[0] * (1 - g1_k_grid[1]) * g1_k_grid[2]
        g1[6] = g1[6] * (1 - g1_k_grid[0]) * g1_k_grid[1] * g1_k_grid[2]
        g1[7] = g1[7] * g1_k_grid[0] * g1_k_grid[1] * g1_k_grid[2]

    return *g0, *g1, pe


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

# 今までの総合

from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import random
import datetime

from torch import dtype

from utils import *
from models import *
from fp_def import *
from var2 import *

print_(datetime.datetime.now(), PRINTLOG_PATH)

for var in over_write_variable_dict.keys():
    print_(f"{var} : {eval(var)}", PRINTLOG_PATH)


def random_crop_dataset(datasets, crop_size, num_crops, uniform_distribution, dim=2):
    crops = []
    coords = []
    if uniform_distribution:
        lod = random.randint(0, MAX_MIP_LEVEL)
    else:
        lod = int(math.floor(-math.log2(random.random()) / 2))
        if lod > MAX_MIP_LEVEL:
            lod = MAX_MIP_LEVEL

    dataset = datasets[lod]
    data_size = dataset.shape[1]
    re_crop_size = max(1, crop_size // pow(2, lod))
    for _ in range(num_crops):
        s = (0, data_size - re_crop_size + 1)
        start_coord = torch.randint(s[0], s[1], (dim,)).to(DEVICE)
        end_coord = start_coord + re_crop_size
        if dim == 2:
            crop = dataset[:, start_coord[0]:end_coord[0], start_coord[1]:end_coord[1]]
        elif dim == 3:
            crop = dataset[:, start_coord[0]:end_coord[0], start_coord[1]:end_coord[1], start_coord[2]:end_coord[2]]
        crop = crop.reshape(3, -1).T
        crops.append(crop)  # チャンネルを最後から最初に移動
        coords.append(start_coord)
    return torch.stack(crops), torch.stack(coords), lod


# フルカラー画像用デコーダの定義
class ColorDecoder(nn.Module):
    def __init__(self):
        super(ColorDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(DECODER_INPUT_CHANNEL, HIDDEN_LAYER_CHANNEL),
            nn.GELU(),
            nn.Linear(HIDDEN_LAYER_CHANNEL, HIDDEN_LAYER_CHANNEL),
            nn.GELU(),
            nn.Linear(HIDDEN_LAYER_CHANNEL, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


def add_noise_for_decoder_input(decoder_input):
    fp_g0_c = FEATURE_PYRAMID_G0_CHANNEL
    fp_g1_c = FEATURE_PYRAMID_G1_CHANNEL
    if COMPRESSION_METHOD == 3:
        g0_k = 8
    else:
        g0_k = 4
    decoder_input[0:g0_k * fp_g0_c] += (torch.rand_like(decoder_input[0:g0_k * fp_g0_c]) - 0.5) / pow(2, FP_G0_BIT)
    decoder_input[g0_k * fp_g0_c:g0_k * fp_g0_c + fp_g1_c] += (torch.rand_like(
        decoder_input[g0_k * fp_g0_c:g0_k * fp_g0_c + fp_g1_c]) - 0.5) / pow(2, FP_G1_BIT)
    return decoder_input


def add_noise_to_grid(g, bit):
    g += (torch.rand_like(g) - 0.5) / pow(2, bit)
    return g


def add_noise_to_tuple(gs, bit):
    return tuple(
        g + (torch.rand_like(g) - 0.5) / pow(2, bit) for g in gs
    )


def create_decoder_input(fp, coords, num_crop, mip_level, add_noise, image_size=0):
    decoder_input = []
    fl = feature_pyramid_mip_levels_dict[mip_level]
    if image_size != 0:
        sample_number = image_size
    else:
        sample_number = pow(2, max(0, CROP_MIP_LEVEL - mip_level))
    step_number = pow(2, mip_level - FEATURE_PYRAMID_SIZE_RATE - fl * 2)
    pe_step_number = pow(2, mip_level)
    sample_ranges = torch.arange(sample_number, dtype=MLP_DTYPE, device=DEVICE).repeat(FP_DIMENSION, 1)
    lod_tensor = torch.ones(1, pow(sample_number, FP_DIMENSION), dtype=MLP_DTYPE, device=DEVICE)
    for i in range(num_crop):
        g0s, g1s, pe = create_g0_g1_pe(fp, fl, coords[i].view(FP_DIMENSION, 1), step_number, pe_step_number,
                                     sample_ranges, PE_CHANNEL, COMPRESSION_METHOD, DEVICE, MLP_DTYPE)
        g0s = add_noise_to_tuple(g0s, FP_G0_BIT)
        g1s = add_noise_to_tuple(g1s, FP_G1_BIT)
        g1 = create_sum_g1(g1s, sample_ranges, coords[i].view(FP_DIMENSION, 1), step_number, COMPRESSION_METHOD)
        decoder_input.append(torch.cat([*g0s, g1, pe, lod_tensor * mip_level], dim=0))
    decoder_input = torch.cat(decoder_input, dim=1)
    if add_noise:
        decoder_input = add_noise_for_decoder_input(decoder_input)
    return decoder_input.T


# モデルの学習
def train_models(fp):
    # 単一の画像での訓練ループ
    accumulator = 0.0
    judge_freeze = True
    start_interval_time = time.perf_counter()
    for epoch in range(NUM_EPOCH):
        # print(epoch)
        accumulator += UNIFORM_DISTRIBUTION_RATE
        if accumulator >= 1.0:
            accumulator -= 1.0
            uniform_distribution = True
        else:
            uniform_distribution = False
        if epoch > NUM_EPOCH * 0.95:
            if judge_freeze:
                fp_freeze(fp)
                before_fp = fp
                fp = fp_all_quantize(fp, FP_G0_BIT, FP_G1_BIT)
                diff_fps(before_fp, fp, PRINTLOG_PATH)
                judge_freeze = False
        start_epoch_time = time.perf_counter()
        inputs, coords, lod = random_crop_dataset(images, CROP_SIZE, NUM_CROP, uniform_distribution, dim=FP_DIMENSION)
        # print(coord)
        # print(convert_coordinate_start(device, coord, 8, 8))
        fl = feature_pyramid_mip_levels_dict[lod]
        # print(lod, fl)
        if epoch < NUM_EPOCH * 0.95:
            add_noise = True
        else:
            add_noise = False

        decoder_input = create_decoder_input(fp, coords, NUM_CROP, lod, add_noise)

        decoder_output = decoder(decoder_input)
        target = inputs.reshape(-1, 3)
        loss = criterion(decoder_output, target)
        if TF_WRITE_PSNR:
            psnr = calculate_psnr(quantize_from_norm_to_bit(decoder_output, OUTPUT_BIT),
                                  quantize_from_norm_to_bit(target, OUTPUT_BIT))

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        fp_quantize_clamp(fp, fl, FP_G0_BIT, FP_G1_BIT)

        end_epoch_time = time.perf_counter()

        elapsed_time = end_epoch_time - start_epoch_time

        writer.add_scalar('Loss/train_epoch_label', loss.item(), epoch + 1)
        if TF_WRITE_TIME:
            writer.add_scalar('Time/epoch_label', elapsed_time, epoch + 1)
        if TF_WRITE_PSNR:
            writer.add_scalar('PSNR/epoch', psnr, epoch + 1)

        if (epoch + 1) % INTERVAL_PRINT == 0:
            printlog = f'Epoch [{epoch + 1}/{NUM_EPOCH}]'
            if TF_PRINT_LOSS:
                printlog += f' Loss: {loss.item():.4f}'
            if TF_PRINT_PSNR:
                reconstructed = decode_image(fp_all_quantize(fp, FP_G0_BIT, FP_G1_BIT), decoder, 0, False)
                if FP_DIMENSION == 2:
                    all_psnr = calculate_psnr(
                        quantize_from_norm_to_bit(reconstructed, OUTPUT_BIT),
                        quantize_from_norm_to_bit(images[0].permute(1, 2, 0), OUTPUT_BIT))
                elif FP_DIMENSION == 3:
                    all_psnr = calculate_psnr(
                        quantize_from_norm_to_bit(reconstructed, OUTPUT_BIT),
                        quantize_from_norm_to_bit(images[0].permute(1, 2, 3, 0), OUTPUT_BIT))
                writer.add_scalar('PSNR/mip0', all_psnr, epoch + 1)
                printlog += f' PSNR: {all_psnr:.4f}'
            if TF_WRITE_TIME:
                end_interval_time = time.perf_counter()
                elapsed_time = end_interval_time - start_interval_time
                printlog += f' Time: {elapsed_time:.4f}'
                start_interval_time = time.perf_counter()
            if TF_PRINT_PSNR or TF_PRINT_LOSS or TF_WRITE_TIME:
                print_(printlog, PRINTLOG_PATH)
            else:
                print(f'Epoch [{epoch + 1}/{NUM_EPOCH}]')

        if (epoch + 1) % INTERVAL_SAVE_MODEL == 0:
            # エンコーダとデコーダの保存
            torch.save(decoder.state_dict(), f'model/{SAVE_NAME}_{epoch}_decoder.pth')


# デコード
def decode_image(fp, arc_decoder, mip_level, pr=True, div_size=10):
    with (torch.no_grad()):
        power = MAX_MIP_LEVEL - mip_level
        div_slice = pow(FP_DIMENSION, max(power - div_size, 0))
        div_count = pow(div_slice, FP_DIMENSION)
        decode_size = IMAGE_SIZE // pow(2, mip_level)
        if div_count == 1:
            coord = torch.zeros((1, FP_DIMENSION), dtype=MLP_DTYPE, device=DEVICE)
            decoder_input = create_decoder_input(fp, coord, 1, mip_level, False, IMAGE_SIZE)
            decoder_output = arc_decoder(decoder_input)
            if pr:
                print_(decoder_output.shape, PRINTLOG_PATH)
            if FP_DIMENSION == 2:
                return decoder_output.reshape(decode_size, decode_size, 3)
            elif FP_DIMENSION == 3:
                return decoder_output.reshape(decode_size, decode_size, decode_size, 3)
        else:
            if FP_DIMENSION == 2:
                result = torch.zeros(IMAGE_SIZE // pow(2, mip_level), IMAGE_SIZE // pow(2, mip_level), 3,
                                     dtype=MLP_DTYPE)
            elif FP_DIMENSION == 3:
                result = torch.zeros(IMAGE_SIZE // pow(2, mip_level), IMAGE_SIZE // pow(2, mip_level),
                                     IMAGE_SIZE // pow(2, mip_level), 3, dtype=MLP_DTYPE)
            for i in range(div_count):
                sample_number = IMAGE_SIZE // pow(2, mip_level + max(power - div_size, 0))
                if FP_DIMENSION == 2:
                    x = (i % div_slice) * sample_number
                    y = (i // div_slice) * sample_number
                    coord = torch.tensor([[x, y]], dtype=MLP_DTYPE, device=DEVICE)
                elif FP_DIMENSION == 3:
                    x = (i % div_slice) * sample_number
                    y = (i // div_slice % div_slice) * sample_number
                    z = (i // div_slice // div_slice) * sample_number
                    coord = torch.tensor([[x, y, z]], dtype=MLP_DTYPE, device=DEVICE)
                decoder_input = create_decoder_input(fp, coord, 1, mip_level, False, sample_number)
                decoder_output = arc_decoder(decoder_input)
                if pr:
                    print_(decoder_output.shape, PRINTLOG_PATH)
                if FP_DIMENSION == 2:
                    result[x:sample_number + x, y:sample_number + y, :] = decoder_output.reshape(sample_number,
                                                                                                 sample_number, 3)
                elif FP_DIMENSION == 3:
                    result[x:sample_number + x, y:sample_number + y, z:sample_number + z, :] = decoder_output.reshape(
                        sample_number, sample_number, sample_number, 3)
            return result


# モデルのインスタンス化
decoder = ColorDecoder().to(DEVICE)
criterion = nn.MSELoss()
feature_pyramid, feature_pyramid_levels = create_pyramid(FEATURE_PYRAMID_SIZE, FP_DIMENSION,
                                                         FEATURE_PYRAMID_G0_CHANNEL, FP_G0_BIT,
                                                         FEATURE_PYRAMID_G1_CHANNEL, FP_G1_BIT,
                                                         DEVICE, MLP_DTYPE, TF_NO_MIP)
for g in feature_pyramid:
    safe_statistics(g, PRINTLOG_PATH)
feature_pyramid_mip_levels_dict = create_pyramid_mip_levels(IMAGE_SIZE, FEATURE_PYRAMID_SIZE_RATE)
optimizer = optim.Adam([
    {'params': feature_pyramid, 'lr': 0.01},  # fpの初期学習率
    {'params': decoder.parameters(), 'lr': 0.005}  # decoderの初期学習率
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCH, eta_min=0)
writer = SummaryWriter(f'../log/{SAVE_NAME}')


# 学習・圧縮・復元の処理
def process_images(train_model, fp):
    if train_model:
        start = time.perf_counter()
        train_models(fp)
        end = time.perf_counter()
        print_("学習時間：" + str(end - start), PRINTLOG_PATH)

        for g in fp:
            safe_statistics(g, PRINTLOG_PATH)

        compressed_fp = fp_savable(fp, FP_G0_BIT, bits2dtype_torch(FP_G0_BIT),
                                   FP_G1_BIT, bits2dtype_torch(FP_G1_BIT))

        # デコーダの保存

        torch.save(decoder.state_dict(), f'model/{SAVE_NAME}_decoder.pth')
        torch.save(decoder.state_dict(), make_filename_by_seq(f'model/{SAVE_NAME}', f'{SAVE_NAME}_decoder.pth'))
        torch.save(compressed_fp, f'feature_pyramid/{SAVE_NAME}_feature_pyramid.pth')
        torch.save(compressed_fp,
                   make_filename_by_seq('feature_pyramid/{save_name}', f'{SAVE_NAME}_feature_pyramid.pth'))

        fp = fp_all_quantize(fp, FP_G0_BIT, FP_G1_BIT)
    else:
        # デコーダの訓練済みパラメータのロード
        decoder.load_state_dict(torch.load(f'model/{SAVE_NAME}_decoder.pth'))
        decoder.eval()
        compressed_fp = torch.load(f'feature_pyramid/{SAVE_NAME}_feature_pyramid.pth')
        fp = fp_load(compressed_fp, FP_G0_BIT, FP_G1_BIT, MLP_DTYPE)

    reconstructed_images = []
    # デコード
    for i in range(MAX_MIP_LEVEL + 1):
        start = time.perf_counter()
        reconstructed = decode_image(fp, decoder, i)
        end = time.perf_counter()
        print_("展開時間：" + str(end - start), PRINTLOG_PATH)

        reconstructed_image = quantize_from_norm_to_bit(reconstructed.cpu().numpy(), OUTPUT_BIT)
        reconstructed_image = reconstructed_image.astype(bits2dtype_np(OUTPUT_BIT))
        reconstructed_images.append(reconstructed_image)

        if IMAGE_DIMENSION == 2:
            reconstructed_image_saved = Image.fromarray(reconstructed_image)
            reconstructed_image_saved.save(make_filename_by_seq(f'image/{SAVE_NAME}', f'{SAVE_NAME}_{i}.png'))
        if COMPRESSION_METHOD == 2:
            size = IMAGE_3D_SIZE
            reconstructed_movie = np.zeros((size, size, size, 3), dtype=np.uint8)
            for x in range(size):
                row = x // (IMAGE_SIZE // size)  # 行の計算
                col = x % (IMAGE_SIZE // size)  # 列の計算
                # print(row * size, (row + 1) * size, col * size, (col + 1) * size)
                reconstructed_movie[x] = reconstructed_images[i][row * size:(row + 1) * size,
                                         col * size:(col + 1) * size, :]

            timelaps(reconstructed_movie, make_filename_by_seq(f'image/{SAVE_NAME}', f'{SAVE_NAME}_{i}.avi'))
        elif COMPRESSION_METHOD == 3 or COMPRESSION_METHOD == 4:
            timelaps(reconstructed_image, make_filename_by_seq(f'image/{SAVE_NAME}', f'{SAVE_NAME}_{i}.avi'))

    return reconstructed_images


if IMAGE_DIMENSION == 2:
    if COMPRESSION_METHOD != 1:
        print_("Error: COMPRESSION_METHOD must be 1 for 2d image", PRINTLOG_PATH)
    if IMAGE_DTYPE == 'image':
        image_original = Image.open(IMAGE_PATH).convert('RGB')
        images = []
        for i in range(MAX_MIP_LEVEL + 1):
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE // pow(2, i), IMAGE_SIZE // pow(2, i))),
                transforms.ToTensor()
            ])
            image = transform(image_original).to(DEVICE)
            print(image.shape)
            images.append(image)
elif IMAGE_DIMENSION == 3:
    if IMAGE_DTYPE == "movie":
        movie_original = readClip(IMAGE_PATH)
    elif IMAGE_DTYPE == "ndarray":
        movie_original = np.load(IMAGE_PATH)

    movie_original = quantize_from_bit_to_bit(movie_original, IMAGE_BIT)

    if COMPRESSION_METHOD == 1:
        print_("Error: COMPRESSION_METHOD must not be 1 for 3d image", PRINTLOG_PATH)
    elif COMPRESSION_METHOD == 2:
        movie = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for i in range(IMAGE_3D_SIZE):
            row = i // (IMAGE_SIZE // IMAGE_3D_SIZE)  # 行の計算
            col = i % (IMAGE_SIZE // IMAGE_3D_SIZE)  # 列の計算
            movie[row * IMAGE_3D_SIZE:(row + 1) * IMAGE_3D_SIZE, col * IMAGE_3D_SIZE:(col + 1) * IMAGE_3D_SIZE, :] = \
                movie_original[i]
        image_original = Image.fromarray(movie, 'RGB')
        images = []
        for i in range(MAX_MIP_LEVEL + 1):
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE // pow(2, i), IMAGE_SIZE // pow(2, i))),
                transforms.ToTensor()
            ])
            image = transform(image_original).to(DEVICE)
            print(image.shape)
            images.append(image)
    elif COMPRESSION_METHOD == 3 or COMPRESSION_METHOD == 4:
        images = []
        for i in range(MAX_MIP_LEVEL + 1):
            # [T, H, W, C] -> [C, T, H, W]
            movie = movie_original.transpose(3, 0, 1, 2)
            image = torch.tensor(movie).float().to(MLP_DTYPE) / pow(2, IMAGE_BIT)
            image = image.to(DEVICE)
            images.append(image)

reconstructed_images = process_images(TF_TRAIN_MODEL, feature_pyramid)
np.set_printoptions(threshold=1000)
"""
print(torch.floor(reconstructed_totyuu * (pow(2, 8) - 1) + 0.5))

reconstructed_totyuu = quantize_from_norm_to_bit(reconstructed_totyuu, 8)

print(reconstructed_totyuu.cpu().numpy())
print(reconstructed_totyuu.cpu().numpy().dtype)

print(reconstructed_totyuu.cpu().numpy().astype(bits2dtype_np(OUTPUT_BITS)).astype(np.float32))

print(reconstructed_images[0].astype(np.float32) - reconstructed_totyuu.cpu().numpy().astype(bits2dtype_np(OUTPUT_BITS)).astype(np.float32))
print(np.mean(reconstructed_images[0].astype(np.float32) - reconstructed_totyuu.cpu().numpy().astype(bits2dtype_np(OUTPUT_BITS)).astype(np.float32)))
print(np.max(reconstructed_images[0].astype(np.float32) - reconstructed_totyuu.cpu().numpy().astype(bits2dtype_np(OUTPUT_BITS)).astype(np.float32)))
print(np.min(reconstructed_images[0].astype(np.float32) - reconstructed_totyuu.cpu().numpy().astype(bits2dtype_np(OUTPUT_BITS)).astype(np.float32)))
print(np.mean(np.abs(reconstructed_images[0].astype(np.float32) - reconstructed_totyuu.cpu().numpy().astype(bits2dtype_np(OUTPUT_BITS)).astype(np.float32))))

a = images[i].cpu().numpy().transpose(1, 2, 3, 0).astype(np.float32) * 255 - quantize_from_norm_to_bit(images[0].permute(1, 2, 3, 0), OUTPUT_BITS).cpu().numpy().astype(np.float32)
print(a)
print(np.mean(a))
print(np.max(a))
print(np.min(a))
print(np.mean(np.abs(a)))
"""

for i in range(MAX_MIP_LEVEL + 1):
    if IMAGE_DIMENSION == 2 or COMPRESSION_METHOD == 2:
        psnr = calculate_psnr(
            quantize_from_norm_to_bit(images[i].cpu().numpy().transpose(1, 2, 0), OUTPUT_BIT),
            reconstructed_images[i].astype(np.float32))
    elif IMAGE_DIMENSION == 3:
        psnr = calculate_psnr(
            quantize_from_norm_to_bit(images[i].cpu().numpy().transpose(1, 2, 3, 0), OUTPUT_BIT),
            reconstructed_images[i].astype(np.float32))
    print_(f"psnr: {psnr}", PRINTLOG_PATH)

# for i in range(MAX_MIP_LEVEL + 1):
#    save_result_to_csv(reconstructed_movie, make_filename_by_seq(f'LUT/{SAVE_NAME}', f'{SAVE_NAME}_{i}.csv'))

if TF_SHOW_RESULT:
    # 画像の表示
    plt.figure(figsize=(6, 3))

    # オリジナル画像
    plt.subplot(1, 2, 1)
    plt.imshow(images[0].cpu().numpy().squeeze().transpose(1, 2, 0))
    plt.axis('off')
    plt.title('Original Image')

    # 再構成された画像
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_images[0])
    plt.axis('off')
    plt.title('Reconstructed Image')

    plt.show()

print_(datetime.datetime.now(), PRINTLOG_PATH)

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
from utils import *
from models import *
from fp_def import *
from var2 import *

print_(datetime.datetime.now(), PRINTLOG_PATH)

for var in over_write_variable_dict.keys():
    print_(f"{var} : {eval(var)}", PRINTLOG_PATH)


def random_crop_dataset(datasets, crop_size, num_crops, uniform_distribution, dim=2):
    crops = []
    coord = []
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
        coord.append(start_coord)
    return torch.stack(crops), torch.stack(coord), lod


# フルカラー画像用デコーダの定義
class ColorDecoder(nn.Module):
    def __init__(self):
        super(ColorDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(DECODER_INPUT_CHANNELS, HIDDEN_LAYER_CHANNELS),
            nn.GELU(),
            nn.Linear(HIDDEN_LAYER_CHANNELS, HIDDEN_LAYER_CHANNELS),
            nn.GELU(),
            nn.Linear(HIDDEN_LAYER_CHANNELS, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


def create_decoder_input_2d(fp, coord, num_crops, fl, mip_level):
    # fp:feature_pyramid
    # fl:feature_level
    # G0:fp[fl*2]
    # G1:fp[fl*2+1]
    decoder_input = []
    irregular = False
    sample_number = pow(2, max(0, 8 - mip_level))
    step_number = pow(2, mip_level - (fl + 1) * 2)
    # print("step_number", step_number)
    # print(mip_level, sample_number, step_number)
    # start = time.perf_counter()

    x_range = torch.arange(sample_number).to(DEVICE)
    y_range = torch.arange(sample_number).to(DEVICE)
    lod_tensor = torch.ones(1, sample_number * sample_number).to(DEVICE)
    # print(x_range)
    # print(coord)

    for i in range(num_crops):
        g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, pe = create_g0_g1(fp, fl, coord[i][0], coord[i][1],
                                                                          step_number, x_range, y_range, PE_CHANNELS,
                                                                          DEVICE, MLP_DTYPE, TF_USE_TRI_PE)
        decoder_input.append(
            torch.cat([g0_0, g0_1, g0_2, g0_3, g1_0 + g1_1 + g1_2 + g1_3, pe, lod_tensor * mip_level], dim=0)
        )
    decoder_input = torch.cat(decoder_input, dim=1)
    # end = time.perf_counter()
    # print(end - start)
    return decoder_input.T


def create_decoder_input_3d(fp, coord, num_crops, fl, mip_level):
    # fp:feature_pyramid
    # fl:feature_level
    # G0:fp[fl*2]
    # G1:fp[fl*2+1]
    decoder_input = []
    irregular = False
    sample_number = pow(2, max(0, CROP_MIP_LEVEL - mip_level))
    step_number = pow(2, mip_level - (fl + 1) * 2)
    # print("step_number", step_number)
    # print(mip_level, sample_number, step_number)
    # start = time.perf_counter()

    x_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    y_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    z_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    lod_tensor = torch.ones(1, pow(sample_number, 3), dtype=MLP_DTYPE).to(DEVICE)
    # print(x_range)
    # print(coord)

    for i in range(num_crops):
        g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7, g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7, pe = (
            create_g0_g1_3d(fp, fl, coord[i][0], coord[i][1], coord[i][2], step_number, x_range, y_range, z_range, PE_CHANNELS, DEVICE, MLP_DTYPE))
        decoder_input.append(
            torch.cat([g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7,
                       g1_0 + g1_1 + g1_2 + g1_3 + g1_4 + g1_5 + g1_6 + g1_7, pe, lod_tensor * mip_level], dim=0)
        )
    # print(g0_0.dtype, g0_1.dtype, g0_2.dtype, g0_3.dtype, g0_4.dtype, g0_5.dtype, g0_6.dtype, g0_7.dtype, g1_0.dtype, g1_1.dtype, g1_2.dtype, g1_3.dtype, g1_4.dtype, g1_5.dtype, g1_6.dtype, g1_7.dtype, pe.dtype)
    decoder_input = torch.cat(decoder_input, dim=1)
    # end = time.perf_counter()
    # print(end - start)
    return decoder_input.T


def create_decoder_input_3d_v2(fp, coord, num_crops, fl, mip_level):
    # fp:feature_pyramid
    # fl:feature_level
    # G0:fp[fl*2]
    # G1:fp[fl*2+1]
    decoder_input = []
    irregular = False
    sample_number = pow(2, max(0, CROP_MIP_LEVEL - mip_level))
    step_number = pow(2, mip_level - (fl + 1) * 2)
    # print("step_number", step_number)
    # print(mip_level, sample_number, step_number)
    # start = time.perf_counter()

    x_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    y_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    z_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    lod_tensor = torch.ones(1, pow(sample_number, 3), dtype=MLP_DTYPE).to(DEVICE)
    # print(x_range)
    # print(coord)

    for i in range(num_crops):
        g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7, pe = (
            create_g0_g1_3d_v2(fp, fl, coord[i][0], coord[i][1], coord[i][2], step_number, x_range, y_range, z_range, PE_CHANNELS, DEVICE, MLP_DTYPE))
        decoder_input.append(
            torch.cat([g0_0, g0_1, g0_2, g0_3,
                       g1_0 + g1_1 + g1_2 + g1_3 + g1_4 + g1_5 + g1_6 + g1_7, pe, lod_tensor * mip_level], dim=0)
        )
    decoder_input = torch.cat(decoder_input, dim=1)
    # end = time.perf_counter()
    # print(end - start)
    return decoder_input.T


def finally_decode_input_2d(fp, image_size, mip_level, x=0, y=0):
    sample_number = image_size
    fl = feature_pyramid_mip_levels_dict[mip_level]
    step_number = pow(2, mip_level - (fl + 1) * 2)
    x_range = torch.arange(sample_number).to(DEVICE)
    y_range = torch.arange(sample_number).to(DEVICE)
    lod_tensor = torch.ones(1, sample_number * sample_number).to(DEVICE)
    g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, pe = create_g0_g1(fp, fl, x, y,
                                                                      step_number, x_range, y_range, PE_CHANNELS,
                                                                      DEVICE, MLP_DTYPE, TF_USE_TRI_PE)
    decoder_input = torch.cat([g0_0, g0_1, g0_2, g0_3, g1_0 + g1_1 + g1_2 + g1_3, pe, lod_tensor * mip_level], dim=0)
    return decoder_input.T


def finally_decode_input_3d(fp, image_size, mip_level, x=0, y=0, z=0):
    sample_number = image_size
    fl = feature_pyramid_mip_levels_dict[mip_level]
    step_number = pow(2, mip_level - (fl + 1) * 2)
    x_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    y_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    z_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    lod_tensor = torch.ones(1, pow(sample_number, 3), dtype=MLP_DTYPE).to(DEVICE)
    g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7, g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7, pe = (
        create_g0_g1_3d(fp, fl, x, y, z, step_number, x_range, y_range, z_range, PE_CHANNELS, DEVICE, MLP_DTYPE))
    decoder_input = torch.cat([g0_0, g0_1, g0_2, g0_3, g0_4, g0_5, g0_6, g0_7,
                       g1_0 + g1_1 + g1_2 + g1_3 + g1_4 + g1_5 + g1_6 + g1_7, pe, lod_tensor * mip_level], dim=0)
    return decoder_input.T


def finally_decode_input_3d_v2(fp, image_size, mip_level, x=0, y=0, z=0):
    sample_number = image_size
    fl = feature_pyramid_mip_levels_dict[mip_level]
    step_number = pow(2, mip_level - (fl + 1) * 2)
    x_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    y_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    z_range = torch.arange(sample_number, dtype=MLP_DTYPE).to(DEVICE)
    lod_tensor = torch.ones(1, pow(sample_number, 3), dtype=MLP_DTYPE).to(DEVICE)
    g0_0, g0_1, g0_2, g0_3, g1_0, g1_1, g1_2, g1_3, g1_4, g1_5, g1_6, g1_7, pe = (
        create_g0_g1_3d_v2(fp, fl, x, y, z, step_number, x_range, y_range, z_range, PE_CHANNELS, DEVICE, MLP_DTYPE))
    decoder_input = torch.cat([g0_0, g0_1, g0_2, g0_3,
                       g1_0 + g1_1 + g1_2 + g1_3 + g1_4 + g1_5 + g1_6 + g1_7, pe, lod_tensor * mip_level], dim=0)
    return decoder_input.T


# モデルの学習
def train_models(fp):
    # 単一の画像での訓練ループ
    count = 0
    accumulator = 0.0
    judge_freeze = True
    for epoch in range(NUM_EPOCHS):
        accumulator += UNIFORM_DISTRIBUTION_RATE
        if accumulator >= 1.0:
            accumulator -= 1.0
            uniform_distribution = True
        else:
            uniform_distribution = False
        if epoch > NUM_EPOCHS * 0.95:
            if judge_freeze:
                fp_freeze(fp)
                fp = fp_all_quantize(fp, FP_BITS)
                judge_freeze = False
        start_epoch_time = time.perf_counter()
        inputs, coord, lod = random_crop_dataset(images, CROP_SIZE, NUM_CROPS, uniform_distribution, dim=FP_DIMENSION)
        # print(coord)
        # print(convert_coordinate_start(device, coord, 8, 8))
        fl = feature_pyramid_mip_levels_dict[lod]
        # print(lod, fl)

        if FP_DIMENSION == 2:
            decoder_input = create_decoder_input_2d(fp, coord, NUM_CROPS, fl, lod)
        elif FP_DIMENSION == 3:
            if COMPRESSION_METHOD == 4:
                decoder_input = create_decoder_input_3d_v2(fp, coord, NUM_CROPS, fl, lod)
            else:
                decoder_input = create_decoder_input_3d(fp, coord, NUM_CROPS, fl, lod)

        # print(torch.any(torch.isinf(decoder_input)))
        if epoch < NUM_EPOCHS * 0.95:
            # 量子化誤差を考慮した一様分布ノイズを生成
            noise = (torch.rand_like(decoder_input) - 0.5) / (2 ** FP_BITS)
            decoder_input_noise = decoder_input + noise
        else:
            # decoder_input_noise = quantize_norm(decoder_input, num_bits)
            decoder_input_noise = decoder_input

        decoder_output = decoder(decoder_input_noise)
        # decoder_output = decoder(decoder_input)
        target = inputs.reshape(-1, 3)
        loss = criterion(decoder_output, target)
        if TF_WRITE_PSNR:
            psnr = calculate_psnr(quantize_to_bit(decoder_output, OUTPUT_BITS), quantize_to_bit(target, OUTPUT_BITS))

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        fp_quantize_clamp(fp, fl, FP_BITS)

        end_epoch_time = time.perf_counter()

        elapsed_time = end_epoch_time - start_epoch_time

        writer.add_scalar('Loss/train_epoch_label', loss.item(), epoch + 1)
        if TF_WRITE_TIME:
            writer.add_scalar('Time/epoch_label', elapsed_time, epoch + 1)
        if TF_WRITE_PSNR:
            writer.add_scalar('PSNR/epoch', psnr, epoch + 1)

        if (epoch + 1) % INTERVAL_PRINT == 0:
            if TF_PRINT_PSNR:
                reconstructed = decode_image(fp_all_quantize(fp, FP_BITS), decoder, i, False)
                if FP_DIMENSION == 2:
                    all_psnr = calculate_psnr(quantize_to_bit(reconstructed, OUTPUT_BITS),
                                            quantize_to_bit(images[0].permute(1, 2, 0), OUTPUT_BITS))
                elif FP_DIMENSION == 3:
                    all_psnr = calculate_psnr(quantize_to_bit(reconstructed, OUTPUT_BITS),
                                              quantize_to_bit(images[0].permute(1, 2, 3, 0), OUTPUT_BITS))
                writer.add_scalar('PSNR/mip0', all_psnr, epoch + 1)
                if TF_PRINT_LOG:
                    print_(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f} PSNR: {all_psnr:.4f}', PRINTLOG_PATH)
                else:
                    print_(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], PSNR: {all_psnr:.4f}',
                           PRINTLOG_PATH)
            elif TF_PRINT_LOG:
                print_(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}', PRINTLOG_PATH)
            else:
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}]')

        if (epoch + 1) % INTERVAL_SAVE_MODEL == 0:
            # エンコーダとデコーダの保存
            torch.save(decoder.state_dict(), f'model/{SAVE_NAME}_{epoch}_decoder.pth')


# デコード
def decode_image(fp, arc_decoder, mip_level, pr=True, div_size=10):
    with (torch.no_grad()):
        power = MAX_MIP_LEVEL - mip_level
        div_slice = pow(2, max(power - div_size, 0))
        div_count = pow(div_slice, 2)
        decode_size = IMAGE_SIZE // pow(2, mip_level)
        if div_count == 1:
            if FP_DIMENSION == 2:
                decoder_input = finally_decode_input_2d(fp, decode_size, mip_level)
            elif FP_DIMENSION == 3:
                if COMPRESSION_METHOD == 4:
                    decoder_input = finally_decode_input_3d_v2(fp, decode_size, mip_level)
                else:
                    decoder_input = finally_decode_input_3d(fp, decode_size, mip_level)
            decoder_output = arc_decoder(decoder_input)
            if pr:
                print_(decoder_output.shape, PRINTLOG_PATH)
            if FP_DIMENSION == 2:
                return decoder_output.reshape(decode_size, decode_size, 3)
            elif FP_DIMENSION == 3:
                return decoder_output.reshape(decode_size, decode_size, decode_size, 3)
        else:
            result = torch.zeros(IMAGE_SIZE // pow(2, mip_level), IMAGE_SIZE // pow(2, mip_level), 3, dtype=MLP_DTYPE)
            for i in range(div_count):
                sample_number = IMAGE_SIZE // pow(2, mip_level + max(power - div_size, 0))
                x = i % div_slice
                y = i // div_slice
                if FP_DIMENSION == 2:
                    decoder_input = finally_decode_input_2d(fp, sample_number, mip_level, sample_number * x, sample_number * y)
                elif FP_DIMENSION == 3:
                    if COMPRESSION_METHOD == 4:
                        decoder_input = finally_decode_input_3d_v2(fp, sample_number, mip_level, sample_number * x, sample_number * y)
                    else:
                        decoder_input = finally_decode_input_3d(fp, sample_number, mip_level, sample_number * x, sample_number * y)
                decoder_output = arc_decoder(decoder_input)
                if pr:
                    print_(decoder_output.shape, PRINTLOG_PATH)
                result[sample_number * x:sample_number * (x + 1), sample_number * y:sample_number * (y + 1),
                :] = decoder_output.reshape(sample_number, sample_number, 3)
            return result


# モデルのインスタンス化
decoder = ColorDecoder().to(DEVICE)
criterion = nn.MSELoss()
if FP_DIMENSION == 2:
    feature_pyramid, feature_pyramid_levels = create_pyramid(FEATURE_PYRAMID_SIZE, FEATURE_PYRAMID_CHANNELS, FP_BITS,
                                                             DEVICE, MLP_DTYPE, TF_NO_MIP)
elif FP_DIMENSION == 3:
    feature_pyramid, feature_pyramid_levels = create_pyramid_3d(FEATURE_PYRAMID_SIZE, FEATURE_PYRAMID_CHANNELS, FP_BITS,
                                                                DEVICE, MLP_DTYPE, TF_NO_MIP)
for fp in feature_pyramid:
    safe_statistics(fp, PRINTLOG_PATH)
feature_pyramid_mip_levels_dict = create_pyramid_mip_levels(IMAGE_SIZE, FEATURE_PYRAMID_SIZE)
optimizer = optim.Adam([
    {'params': feature_pyramid, 'lr': 0.01},  # fpの初期学習率
    {'params': decoder.parameters(), 'lr': 0.005}  # decoderの初期学習率
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=0)
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

        compressed_fp = fp_savable(fp, FP_BITS, bits2dtype_torch(FP_BITS))

        # デコーダの保存

        torch.save(decoder.state_dict(), f'model/{SAVE_NAME}_decoder.pth')
        torch.save(decoder.state_dict(), make_filename_by_seq(f'model/{SAVE_NAME}', f'{SAVE_NAME}_decoder.pth'))
        torch.save(compressed_fp, f'feature_pyramid/{SAVE_NAME}_feature_pyramid.pth')
        torch.save(compressed_fp,
                   make_filename_by_seq('feature_pyramid/{save_name}', f'{SAVE_NAME}_feature_pyramid.pth'))

        fp = fp_all_quantize(fp, FP_BITS)
    else:
        # デコーダの訓練済みパラメータのロード
        decoder.load_state_dict(torch.load(f'model/{SAVE_NAME}_decoder.pth'))
        decoder.eval()
        compressed_fp = torch.load(f'feature_pyramid/{SAVE_NAME}_feature_pyramid.pth')
        fp = fp_load(compressed_fp, FP_BITS, bits2dtype_torch(FP_BITS))

    reconstructed_images = []
    # デコード
    for i in range(MAX_MIP_LEVEL + 1):
        start = time.perf_counter()
        reconstructed = decode_image(fp, decoder, i)
        end = time.perf_counter()
        print_("展開時間：" + str(end - start), PRINTLOG_PATH)

        reconstructed_image = quantize_to_bit(reconstructed.cpu().numpy(), OUTPUT_BITS)
        reconstructed_image = reconstructed_image.astype(bits2dtype_np(OUTPUT_BITS))
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
                reconstructed_movie[x] = reconstructed_images[0][row * size:(row + 1) * size,
                                         col * size:(col + 1) * size, :]

            timelaps(reconstructed_movie, make_filename_by_seq(f'image/{SAVE_NAME}', f'{SAVE_NAME}_0.avi'))
        elif COMPRESSION_METHOD == 3 or COMPRESSION_METHOD == 4:
            timelaps(reconstructed_image, make_filename_by_seq(f'image/{SAVE_NAME}', f'{SAVE_NAME}_0.avi'))

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

    movie_original = quantize_from_bit_to_bit(movie_original, IMAGE_BITS)

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
            image = torch.tensor(movie).float().to(MLP_DTYPE) / pow(2, IMAGE_BITS)
            image = image.to(DEVICE)
            images.append(image)



reconstructed_images = process_images(TF_TRAIN_MODEL, feature_pyramid)
for i in range(MAX_MIP_LEVEL + 1):
    if IMAGE_DIMENSION == 2:
        psnr = calculate_psnr(images[i].cpu().numpy().transpose(1, 2, 0).astype(np.float32) * 255,
                              reconstructed_images[i].astype(np.float32))
    elif IMAGE_DIMENSION == 3:
        psnr = calculate_psnr(images[i].cpu().numpy().transpose(1, 2, 3, 0).astype(np.float32) * 255,
                              reconstructed_images[i].astype(np.float32))
    print_(f"psnr: {psnr}", PRINTLOG_PATH)

#for i in range(MAX_MIP_LEVEL + 1):
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

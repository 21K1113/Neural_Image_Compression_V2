import torch
import time


def add_noise_to_tuple(tensors, scale_factor):
    return tuple(
        tensor + (torch.rand_like(tensor) - 0.5) / scale_factor for tensor in tensors
    )


def add_noise_to_tensor(tensors, scale_factor):
    stacked = torch.stack(tensors)  # (N, 8, 32768) のテンソルを作成
    noisy = stacked + (torch.rand_like(stacked) - 0.5) / scale_factor
    return noisy


# サンプルデータ

scale_factor = 16

# タプルのまま処理
start = time.time()
for i in range(100):
    tensors = [torch.randn(8, 32768) for _ in range(4)]
    noisy_tuple = add_noise_to_tuple(tensors, scale_factor)
print("タプル処理時間:", time.time() - start)

# テンソルに変換して処理
start = time.time()
for i in range(100):
    tensors = [torch.randn(8, 32768) for _ in range(4)]
    noisy_tensor = add_noise_to_tensor(tensors, scale_factor)
print("テンソル処理時間:", time.time() - start)

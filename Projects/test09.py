import torch
import time

# 大きなテンソルを作成
tensor = torch.rand(1000000) * 1000

# floorを使った場合の計測
start_time = time.time()
fractional_part_floor = tensor - torch.floor(tensor)
print(f"Floor method: {time.time() - start_time} seconds")

# truncを使った場合の計測
start_time = time.time()
fractional_part_trunc = tensor - tensor.trunc()
print(f"Trunc method: {time.time() - start_time} seconds")
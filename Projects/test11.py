import torch


def safe_statistics(tensor):
    # NaNとInfを除外するために、マスクを作成
    valid_mask = torch.isfinite(tensor)

    # 有効な値のみを取得
    valid_tensor = tensor[valid_mask]

    if valid_tensor.numel() == 0:
        print("No valid numbers in the tensor.")
    else:
        # 最大値
        max_val = torch.max(valid_tensor)
        print(f"Max: {max_val.item()}")

        # 最小値
        min_val = torch.min(valid_tensor)
        print(f"Min: {min_val.item()}")

        # 平均
        mean_val = torch.mean(valid_tensor)
        print(f"Mean: {mean_val.item()}")

        # 分散
        var_val = torch.var(valid_tensor)
        print(f"Variance: {var_val.item()}")

    # NaNの有無
    has_nan = torch.isnan(tensor).any()
    print(f"Contains NaN: {has_nan.item()}")

    # Infの有無
    has_inf = torch.isinf(tensor).any()
    print(f"Contains Inf: {has_inf.item()}")


# テスト用のテンソル
tensor = torch.tensor([1.0, 2.0, 3.0, float('nan'), float('inf'), 4.0, -1.0])

safe_statistics(tensor)
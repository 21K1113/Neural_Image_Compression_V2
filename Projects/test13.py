import torch

# 学習後に保存したパラメータを読み込む
final_params1 = torch.load('final_decoder_params.pth')

final_params2 = torch.load(f'model/sample23-3_cuda_Multilayer_para3_64.npy_32_True_1000_8_decoder.pth')

for name in final_params1:
    final_param = final_params1[name]
    post_param = final_params2[name]
    difference = (final_param - post_param).abs().mean()  # 平均絶対差
    print(f"Parameter: {name}, Difference: {difference.item()}")

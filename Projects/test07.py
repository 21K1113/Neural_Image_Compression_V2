import torch


def create_decoder_input(fp, coord, fl, mip_level):
    # fp:feature_pyramid
    # fl:feature_level
    # G0:fp[fl*2]
    # G1:fp[fl*2+1]
    xy_pairs_list = []
    g0_list = []
    g1_list = []
    pow_number = 8 - mip_level
    if pow_number < 0:
        pow_number = 0
    sample_number = pow(2, pow_number)
    print(sample_number)
    step_number = pow(2, mip_level - (fl + 1) * 2)
    print(step_number)
    for y in coord[0]:
        for x in coord[1]:
            x_tensor = torch.floor(torch.arange(x, x + sample_number, 1) * step_number).to(torch.int)
            y_tensor = torch.floor(torch.arange(y, y + sample_number, 1) * step_number).to(torch.int)
            x_grid, y_grid = torch.meshgrid(x_tensor, y_tensor, indexing='ij')
            x_flat = x_grid.reshape(-1)
            y_flat = y_grid.reshape(-1)
            xy_pairs = torch.stack([x_flat, y_flat], dim=1)
            x_indices = xy_pairs[:, 0]
            y_indices = xy_pairs[:, 1]
            print(x_indices)
            g0 = fp[fl * 2][:, x_indices, y_indices]
            g1 = fp[fl * 2][:, x_indices, y_indices]
            g0_list.append(g0)
    decoder_input = torch.cat(g0_list)

    return decoder_input


fp = torch.tensor([[3, 4], [5, 6]])
fp = [fp.reshape([1, 2, 2])]
coord = [torch.tensor([0, 1]), torch.tensor([1, 0])]
fl = 0
mip_level = 0

create_decoder_input(fp, coord, fl, mip_level)

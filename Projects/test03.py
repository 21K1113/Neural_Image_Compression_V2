from collections import defaultdict


def return_2_power(base_size):
    count = 0
    x = base_size
    while x != 1:
        x = x // 2
        count += 1
    return count


def return_pyramid_levels(base_size):
    count = return_2_power(base_size)
    return (count + 1) // 2


def create_pyramid_mip_levels(image_size, base_size):
    count = return_2_power(image_size)
    feature_pyramid_dict = defaultdict(int)
    levels = return_pyramid_levels(base_size)
    for i in range(count+1):
        feature_pyramid_dict[i] = (i // 2) - 1
        if feature_pyramid_dict[i] < 0:
            feature_pyramid_dict[i] = 0
        elif feature_pyramid_dict[i] >= levels:
            feature_pyramid_dict[i] = levels - 1
    return feature_pyramid_dict

print(return_pyramid_levels(128))
print(return_2_power(1024))
print(create_pyramid_mip_levels(1024, 256))
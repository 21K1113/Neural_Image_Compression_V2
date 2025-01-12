import numpy as np
import models

a = np.load("data/Multilayer_para3_256.npy")

b = models.quantize_from_bit_to_bit(a, 8).astype(np.uint8)

np.save("data/quantized_Multilayer_para3_256.npy", b)


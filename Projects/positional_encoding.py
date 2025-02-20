import torch
import torch.nn as nn
from utils import convert_coordinate_start


def tri(x, offset=0.5):
    return 2 * torch.abs((x-offset) % 2 - 1) - 1


# (1 freq, 0 offset), (2 freq, 0.5 offset), (2 freq, 0 offset), (4 freq, 0.5 offset), (4 freq, 0 offset)
class TriangularPositionalEncoding1D(nn.Module):
    def __init__(self, sequence_length=8, octaves=3, include_constant=True):
        super().__init__()

        self.include_constant = include_constant
        self.octaves = octaves
        self.sequence_length = sequence_length
        encodings = []
        # I am basing the positional encoding config based on Fig 5 in the paper
        # Authors don't seem to detail the exact config, and closest thing is the
        # pixel values from the paper
        x = torch.arange(0, sequence_length, step=1)
        for octave in range(octaves):
            div = 2**(octave)
            for i, offset in enumerate((.0, 0.5)):
                if octave == 0 and i == 1:
                    # Skip the second offset in the first octave
                    continue
                encoding = tri(x / (div), offset=offset)
                encodings.append(encoding)
        if include_constant:
            encodings.append(torch.zeros(sequence_length, dtype=encodings[-1].dtype))
        encodings = torch.stack(encodings)
        self.register_buffer('encodings', encodings)

    def forward(self, coordinates):
        b, seq_len = coordinates.shape
        d1, d2 = self.encodings.shape
        encodings = self.encodings.unsqueeze(0).expand(b, d1, d2)
        coordinates = (coordinates % self.sequence_length).unsqueeze(1).expand(b, d1, seq_len)
        results = torch.gather(encodings, 2, coordinates)
        return results


class TriangularPositionalEncoding2D(nn.Module):
    def __init__(self, sequence_length=8, octaves=3, include_constant=True):
        super().__init__()

        self.include_constant = include_constant
        self.octaves = octaves
        self.sequence_length = sequence_length
        self.encoding = TriangularPositionalEncoding1D(sequence_length, octaves, include_constant)

    def forward(self, coordinates, h, w, stride=1):
        full_x, full_y = convert_coordinate_start(coordinates, h, w, stride)
        b = coordinates.shape[0]
        encoding_x = self.encoding(full_x).view(b, -1, h, w)
        encoding_y = self.encoding(full_y).view(b, -1, h, w)
        return torch.cat([encoding_x, encoding_y], dim=1)























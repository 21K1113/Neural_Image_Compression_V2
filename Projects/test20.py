import numpy as np

def bilinear_interpolation_with_center(image, scale, x, y):
    """
    Perform bilinear interpolation considering pixel centers.

    Parameters:
    image (numpy.ndarray): Input image as a 2D array.
    scale (float): Scaling factor for the image.
    x (float): x-coordinate in the scaled image.
    y (float): y-coordinate in the scaled image.

    Returns:
    float: Interpolated pixel value.
    """
    # Original image dimensions
    orig_height, orig_width = image.shape

    # Compute coordinates in the original image (adjusting for pixel centers)
    orig_x = (x + 0.5) / scale - 0.5
    orig_y = (y + 0.5) / scale - 0.5
    print(orig_x, orig_y)

    # Find the integer coordinates surrounding the original coordinates
    x0 = int(np.floor(orig_x))
    x1 = min(x0 + 1, orig_width - 1)
    y0 = int(np.floor(orig_y))
    y1 = min(y0 + 1, orig_height - 1)

    # Ensure coordinates are within bounds
    x0 = max(x0, 0)
    y0 = max(y0, 0)

    # Compute the fractional part
    dx = orig_x - x0
    dy = orig_y - y0

    # Retrieve pixel values
    Q11 = image[y0, x0]
    Q21 = image[y0, x1]
    Q12 = image[y1, x0]
    Q22 = image[y1, x1]

    # Perform bilinear interpolation
    R1 = (1 - dx) * Q11 + dx * Q21
    R2 = (1 - dx) * Q12 + dx * Q22
    P = (1 - dy) * R1 + dy * R2

    return P

# Example usage
if __name__ == "__main__":
    # Example image (grayscale)
    image = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])

    scale = 1  # Scaling factor
    x, y = 0, 0  # Coordinates in the scaled image

    value = bilinear_interpolation_with_center(image, scale, x, y)
    print(f"Interpolated value at ({x}, {y}): {value}")

# some functions are used in the main program

import cv2
from datetime import datetime
import numpy as np


def measure_time(func, *args, **kwargs):
    start_time = datetime.now()
    result = func(*args, **kwargs)
    end_time = datetime.now()
    print(f'Time taken to {func.__name__}: {(end_time - start_time).total_seconds()} seconds')
    return result

def normalize_matrix(matrix):
    min_vals = np.min(matrix, axis=0, keepdims=True)
    max_vals = np.max(matrix, axis=0, keepdims=True)
    return (matrix - min_vals) / (max_vals - min_vals)

def is_power_of_two(n):
    return (n != 0) and (n & (n-1) == 0)


def image_to_optical_blocks(img, block_sizes):

    block_height, block_width = block_sizes[0], block_sizes[1]

    if not is_power_of_two(block_height):
         raise ValueError("Block height must be a power of 2")
    if not is_power_of_two(block_width):
         raise ValueError("Block width must be a power of 2")
    
    if img.ndim == 2:
        img = img[..., np.newaxis]
    
    height, width, channels = img.shape

    # Ensure the image dimensions are multiples of the optical field size
    new_height = height if height % block_height == 0 else height + block_height - height % block_height
    new_width = width if width % block_width == 0 else width + block_width - width % block_width
    
    # Resize the image to fit the optical field size
    img = cv2.resize(img, (new_width, new_height))

    # Reshape to the desired shape
    image_reshaped = img.reshape(new_height // block_height, 
                                block_height,
                                new_width // block_width,
                                block_width,
                                channels
                                ).transpose(0, 2, 1, 3, 4)

    return image_reshaped




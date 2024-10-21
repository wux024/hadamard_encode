import cv2
from spi import image_to_optical_blocks, HadamardTransform, HadamardTransformExtended, measure_time
try:
    import cupy as np
    if np.cuda.runtime.getDeviceCount() > 0:
        print('CUDA device found and using CuPy (GPU)')
    else:
        raise ImportError('No CUDA device found')
except ImportError as e:
    import numpy as np
    print('No CUDA device found and using NumPy (CPU)')


if __name__ == '__main__':
    
    optical_field_size = 128

    hadamard_transform1 = measure_time(HadamardTransform, optical_field_size)

    hadamard_transform2 = measure_time(HadamardTransformExtended, optical_field_size)





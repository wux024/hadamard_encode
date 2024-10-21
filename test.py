import torch
from spi import HadamardTransform, HadamardTransformExtended, measure_time


if __name__ == '__main__':
    
    optical_field_size = 128

    hadamard_transform1 = measure_time(HadamardTransform, optical_field_size, is_torch=False)

    hadamard_transform2 = measure_time(HadamardTransformExtended, optical_field_size, is_torch=False)






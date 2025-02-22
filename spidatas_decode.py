import numpy as np
import h5py
import os
import cv2
from spi import HadamardTransform

optical_field_size = 128
sub_optical_field_sizes = [128, 64, 32, 16]
data_path = 'spidatas/realworldperson/train'
hadamard_transform = HadamardTransform(optical_field_size=optical_field_size)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_and_save_image(data, sub_optical_field_size, file_name, output_type, seed=None, split=False):
    if output_type == 'aliasing':
        result = hadamard_transform.sub_inverse_transform(data, sub_optical_field_size)
        result = result.reshape(optical_field_size, optical_field_size)
        result[:,0] = result[:,1]
        if split:
            if sub_optical_field_size < optical_field_size:
                split_size = (optical_field_size // sub_optical_field_size) ** 2
                result = result[:optical_field_size // split_size, :]
            else:
                return
    elif output_type == 'inverse' or output_type == 'normal':
        sub_hadamard_result = hadamard_transform.extract_submatrix(data, sub_optical_field_size, inverse=output_type=='inverse')
        result = sub_hadamard_result.reshape(sub_optical_field_size, sub_optical_field_size)
    elif output_type == 'hadamard-seed':
        np.random.seed(seed)
        data = np.random.permutation(data)
        sub_hadamard_result = hadamard_transform.extract_submatrix(data, sub_optical_field_size, inverse=False)
        result = sub_hadamard_result.reshape(sub_optical_field_size, sub_optical_field_size)
    else:
        raise ValueError(f'Invalid output_type: {output_type}')
    
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    result = cv2.resize(result, (377, 377))
    
    if output_type == 'normal':
        folder_name = f'images-{optical_field_size}x{optical_field_size}-{sub_optical_field_size}x{sub_optical_field_size}'
    elif output_type == 'hadamard-seed':
        folder_name = f'images-{optical_field_size}x{optical_field_size}-{sub_optical_field_size}x{sub_optical_field_size}-{seed}'
    else:
        folder_name = f'images-{optical_field_size}x{optical_field_size}-{sub_optical_field_size}x{sub_optical_field_size}-{output_type}'
    
    sub_file_name = f'{folder_name}/train/{file_name}.bmp'
    ensure_directory_exists(os.path.dirname(sub_file_name))
    cv2.imwrite(sub_file_name, result)

for file_name in os.listdir(data_path):
    file_path = os.path.join(data_path, file_name)
    base_name = os.path.splitext(file_name)[0]
    with h5py.File(file_path, 'r') as f:
        data_key = list(f.keys())[0]
        data = f[data_key][()].T
    
    for sub_optical_field_size in sub_optical_field_sizes:
        # process_and_save_image(data, sub_optical_field_size, base_name, 'normal')
        # process_and_save_image(data, sub_optical_field_size, base_name, 'inverse')
        process_and_save_image(data, sub_optical_field_size, base_name, 'aliasing', split=True)
        # process_and_save_image(data, sub_optical_field_size, base_name, 'hadamard-seed', 20241215)
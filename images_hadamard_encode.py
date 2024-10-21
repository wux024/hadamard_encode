# This code is used to encode images using Hadamard transform.

import cv2
import argparse
import os

from spi import HadamardTransform, HadamardTransformExtended
from spi import image_to_optical_blocks, is_power_of_two

try:
    import cupy as np
    if np.cuda.runtime.getDeviceCount() > 0:
        print('CUDA device found and using CuPy (GPU)')
    else:
        raise ImportError('No CUDA device found')
except ImportError as e:
    import numpy as np
    print('No CUDA device found and using NumPy (CPU)')


def check_path(dataset_input_base_path, split, image_name, optical_field_size=None, sub_optical_field_size=None, window_size=None, seed=None):
    save_path = os.path.join(dataset_input_base_path)
    if optical_field_size is not None:
        save_path = os.path.join(save_path, f'images-{optical_field_size}x{optical_field_size}')
    
    if sub_optical_field_size is not None:
        save_path += f'-{sub_optical_field_size}x{sub_optical_field_size}'
    
    if window_size is not None:
        save_path += f'-{window_size[0]}x{window_size[1]}'
    
    if seed is not None:
        save_path += f'-{seed}'

    if split is not None:
        save_path = os.path.join(save_path, split)
    else:
        raise ValueError('Split is not specified')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, image_name)
    return save_path

def parse_args():
    parser = argparse.ArgumentParser(description='Hadamard transform for image encoding')
    
    parser.add_argument('--dataset', type=str, default='mouse', help='dataset name')
    parser.add_argument('--optical_field_sizes', type=int, nargs='+', default=[128], help='optical field size (can be multiple values separated by spaces)')
    parser.add_argument('--sub_optical_field_sizes', type=int, nargs='+', default=[32], help='sub-optical field size (can be multiple values separated by spaces)')
    parser.add_argument('--window_size', type=int, nargs='+', default=None, help='window size for Extended Hadamard transform')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    parser.add_argument('--imgsz', type=int, default=None, help='process only a specific image size')

    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()

    dataset = args.dataset
    optical_field_sizes = args.optical_field_sizes
    sub_optical_field_sizes = args.sub_optical_field_sizes
    window_size = args.window_size
    seed = args.seed
    imgsz = args.imgsz

    for optical_field_size in optical_field_sizes:
        if not is_power_of_two(optical_field_size):
            raise ValueError('Optical field size must be a power of 2')
        
    for sub_optical_field_size in sub_optical_field_sizes:
        if not is_power_of_two(sub_optical_field_size):
            raise ValueError('Sub-optical field size must be a power of 2')
    
    dataset_input_base_path = os.path.join('datasets', dataset)

    if not os.path.exists(dataset_input_base_path):
        raise ValueError(f'Dataset {dataset} does not exist')

    dataset_split = ['train', 'val', 'test']

    if window_size is not None:
        hadamard_transform = HadamardTransformExtended()
    else:
        hadamard_transform = HadamardTransform()

    for optical_field_size in optical_field_sizes:

        # set the optical field size for the Hadamard transform
        hadamard_transform.optical_field_size = optical_field_size
        if window_size is not None:
            hadamard_transform.window_size = window_size
            block_sizes = [optical_field_size * window_size[0], optical_field_size * window_size[1]]
        else:
            block_sizes = [optical_field_size, optical_field_size]
        
        if seed is not None:
            hadamard_transform.hadmard_matrix_random(seed)
        
        for split in dataset_split:

            dataset_input_path = os.path.join(dataset_input_base_path, 'images', split)

            image_names = os.listdir(dataset_input_path)

            for image_name in image_names:

                image_path = os.path.join(dataset_input_path, image_name)

                image = cv2.imread(image_path)

                original_height, original_width, _ = image.shape

                if imgsz is not None:
                    image = cv2.resize(image, (imgsz, imgsz))

                image = image.astype(np.float32) / 255.0

                if image is None:
                    print(f'Failed to read image {image_path}')
                    continue

                # make images to blocks
                image_blocks = image_to_optical_blocks(image, block_sizes)

                # sum the blocks along the optical field dimensions
                image_sum = np.sum(image_blocks, axis=(0, 1))

                # normalize the summed image
                save_image_sum = cv2.normalize(image_sum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                # save the summed image
                sum_image_path = check_path(dataset_input_base_path=dataset_input_base_path, 
                                               image_name=image_name, 
                                               split=split, 
                                               optical_field_size=optical_field_size, 
                                               window_size=window_size, 
                                               seed=seed)
                if imgsz is not None:
                    save_image_sum = cv2.resize(save_image_sum, (original_width, original_height))
                cv2.imwrite(sum_image_path, save_image_sum)

                # image channels
                channels = image_blocks.shape[-1]

                # Reshape the image to a 2D array
                image_reshaped = image_sum.reshape(-1, channels)

                # Apply Hadamard transform to the summed image
                hadamard_result = hadamard_transform.transform(image_reshaped)
                saved_hadamard_result = hadamard_result.reshape(optical_field_size, 
                                                                optical_field_size,
                                                                channels)

                hadamard_result_path = check_path(dataset_input_base_path=dataset_input_base_path, 
                                                   split=split, 
                                                   image_name=image_name, 
                                                   optical_field_size=optical_field_size, 
                                                   sub_optical_field_size=optical_field_size, 
                                                   window_size=window_size, 
                                                   seed=seed)
                # normalize the Hadamard result
                saved_hadamard_result = cv2.normalize(saved_hadamard_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                if imgsz is not None:
                    saved_hadamard_result = cv2.resize(saved_hadamard_result, (original_width, original_height))
                cv2.imwrite(hadamard_result_path, saved_hadamard_result)

                for sub_optical_field_size in sub_optical_field_sizes:
                    if sub_optical_field_size >= optical_field_size:
                        continue
                    sub_hadamard_result = hadamard_transform.extract_submatrix(hadamard_result, 
                                                                            sub_optical_field_size)
                    sub_hadamard_result = sub_hadamard_result.reshape(sub_optical_field_size, 
                                                                          sub_optical_field_size, 
                                                                          channels)
                    sub_hadamard_result_path = check_path(dataset_input_base_path=dataset_input_base_path, 
                                                           split=split, 
                                                           image_name=image_name, 
                                                           optical_field_size=optical_field_size, 
                                                           sub_optical_field_size=sub_optical_field_size, 
                                                           window_size=window_size, 
                                                           seed=seed)
                    # normalize the sub-Hadamard result
                    sub_hadamard_result = cv2.normalize(sub_hadamard_result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    if imgsz is not None:
                        sub_hadamard_result = cv2.resize(sub_hadamard_result, (original_width, original_height))
                    cv2.imwrite(sub_hadamard_result_path, sub_hadamard_result)


if __name__ == '__main__':
    main()
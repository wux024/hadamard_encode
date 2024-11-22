import argparse
import os
from spi import HadamardTransform, HadamardTransformExtended, SPIDataloader
from spi import image_to_optical_blocks, is_power_of_two, measure_time
import numpy as np

def parse_args():
    # Create an argument parser for command-line arguments
    parser = argparse.ArgumentParser(description='Hadamard transform for image encoding')
    
    # Add command-line arguments
    parser.add_argument('--dataset', type=str, default='mouse', help='Name of the dataset')
    parser.add_argument('--optical-field-sizes', type=int, nargs='+', default=[128], help='Optical field size (can be multiple values separated by spaces)')
    parser.add_argument('--sub-optical-field-sizes', type=int, nargs='+', default=None, help='Sub-optical field size (can be multiple values separated by spaces)')
    parser.add_argument('--window-size', type=int, nargs='+', default=None, help='Window size for Extended Hadamard transform')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--imgsz', type=int, default=None, help='Process only a specific image size')
    parser.add_argument('--save-aliasing', action='store_true', help='Save the aliasing effect of the Hadamard transform')
    parser.add_argument('--save-hadamard', action='store_true', help='Save the Hadamard transform result')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    return args

def validate_sizes(sizes):
    # Check if the sizes are powers of two
    for size in sizes:
        if not is_power_of_two(size):
            raise ValueError('Size must be a power of 2')

def main():
    # Parse the command-line arguments
    args = parse_args()

    # Extract arguments
    dataset = args.dataset
    optical_field_sizes = args.optical_field_sizes
    sub_optical_field_sizes = args.sub_optical_field_sizes
    window_size = args.window_size
    seed = args.seed
    imgsz = args.imgsz
    save_aliasing = args.save_aliasing
    save_hadamard = args.save_hadamard

    # Validate the sizes
    validate_sizes(optical_field_sizes + sub_optical_field_sizes)
        
    # Construct the dataset input base path
    dataset_input_base_path = os.path.join('datasets', dataset)

    # Check if the dataset directory exists
    if not os.path.exists(dataset_input_base_path):
        raise ValueError('Dataset not found')
    
    # Define the dataset splits (train, val, test)
    dataset_splits = ['train', 'val', 'test']

    # Initialize the dataset loader
    datasetloader = SPIDataloader(dataset_input_base_path=dataset_input_base_path,
                                  window_size=window_size,
                                  aliasing=save_aliasing,
                                  seed=seed,
                                  imgsz=imgsz)

    # Choose the appropriate Hadamard transform class based on the window size
    hadamard_transform = HadamardTransformExtended() if args.window_size else HadamardTransform()
    if seed is not None:
        hadamard_transform.hadmard_matrix_random(seed)
    
    # Iterate over each optical field size
    for optical_field_size in optical_field_sizes:
        print(f'Processing optical field size {optical_field_size} in {dataset} dataset')
        # Set the optical field size for the Hadamard transform
        hadamard_transform.optical_field_size = optical_field_size
        if window_size is not None:
            # Set the window size for the extended Hadamard transform
            hadamard_transform.window_size = window_size
            block_size = [optical_field_size * window_size[0], optical_field_size * window_size[1]]
        else:
            block_size = [optical_field_size, optical_field_size]
        
        # Set a random seed for reproducibility if provided
        if seed is not None:
            hadamard_transform.hadmard_matrix_random(seed)
        
        # Update the dataset loader with the current optical field size
        datasetloader.update_attributes(optical_field_size=optical_field_size)
        
        # Iterate over each dataset split (train, val, test)
        for split in dataset_splits:
            # Update the dataset loader with the current split
            datasetloader.update_attributes(split=split)
            # Load the image paths for the current split
            image_paths = datasetloader.load()
            # Process each image in the current split
            for image_path in image_paths:
                # Get the image name from the path
                image_name = os.path.basename(image_path)
                # Load the image
                image = datasetloader.load_image(image_path)
                # Get the original dimensions of the image
                original_height, original_width, _ = image.shape
                # Preprocess the image
                image = datasetloader.preprocess(image)
                # Convert the image to float32 and normalize to [0, 1]
                image = image.astype(np.float32) / 255.0
                # Split the image into optical blocks
                optical_blocks = image_to_optical_blocks(image, block_size)
                # Compute the sum of the optical blocks
                image_sum = np.sum(optical_blocks, axis=(0, 1))
                # Reshape the image sum for Hadamard transform
                channels = image.shape[-1]
                image_reshaped = image_sum.reshape(-1, channels)
                # Apply the Hadamard transform
                hadamard_result = hadamard_transform.transform(image_reshaped)

                # Iterate over each sub-optical field size
                for sub_optical_field_size in sub_optical_field_sizes:

                    # Inverse the submatrix if requested
                    if save_aliasing and sub_optical_field_size < imgsz:
                        if sub_optical_field_size > optical_field_size:
                            continue
                        elif sub_optical_field_size == optical_field_size:
                            datasetloader.update_attributes(sub_optical_field_size=None, 
                                                        inverse=False, 
                                                        aliasing=False)
                        else:
                            datasetloader.update_attributes(sub_optical_field_size=sub_optical_field_size, 
                                                        aliasing=True, 
                                                        inverse=False)
                        # Compute the aliasing effect of the Hadamard transform
                        sub_aliasing_result = hadamard_transform.sub_inverse_transform(hadamard_result, 
                                                                                       sub_optical_field_size)
                        # Reshape the aliasing effect to the original dimensions
                        sub_aliasing_result = sub_aliasing_result.reshape(optical_field_size, 
                                                                           optical_field_size, 
                                                                           channels)
                        # Normalize and postprocess the aliasing effect
                        sub_aliasing_result = datasetloader.normalize(sub_aliasing_result)
                        sub_aliasing_result = datasetloader.postprocess(sub_aliasing_result, (original_width, original_height))
                        # Build the save path for the aliasing effect
                        sub_aliasing_result_path = datasetloader.build_dataset_path(image_name, original=False)
                        # Save the aliasing effect
                        datasetloader.save(sub_aliasing_result, sub_aliasing_result_path)

                    if save_hadamard:
                        for inverse in [True, False]:
                            # Extract the submatrix from the Hadamard result
                            if sub_optical_field_size > optical_field_size:
                                continue
                            elif sub_optical_field_size == optical_field_size and inverse:
                                continue
                            elif sub_optical_field_size == optical_field_size:
                                sub_hadamard_result = hadamard_result
                                datasetloader.update_attributes(sub_optical_field_size=sub_optical_field_size, 
                                                                inverse=False, 
                                                                aliasing=False)
                            else:
                                sub_hadamard_result = hadamard_transform.extract_submatrix(hadamard_result, 
                                                                                            sub_optical_field_size,
                                                                                            inverse=inverse)
                                datasetloader.update_attributes(sub_optical_field_size=sub_optical_field_size, 
                                                                inverse=inverse, 
                                                                aliasing=False)

                            # Reshape the submatrix to the original dimensions
                            sub_hadamard_result = sub_hadamard_result.reshape(sub_optical_field_size, 
                                                                                sub_optical_field_size, 
                                                                                channels)
                            # Normalize and postprocess the submatrix
                            sub_hadamard_result = datasetloader.normalize(sub_hadamard_result)
                            sub_hadamard_result = datasetloader.postprocess(sub_hadamard_result, (original_width, original_height))
                            # Build the save path for the submatrix
                            sub_hadamard_result_path = datasetloader.build_dataset_path(image_name, original=False)
                            # Save the submatrix
                            datasetloader.save(sub_hadamard_result, sub_hadamard_result_path)


if __name__ == '__main__':
    # Measure the execution time of the main function
    measure_time(main)
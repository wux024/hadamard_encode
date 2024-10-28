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
    parser.add_argument('--optical_field_sizes', type=int, nargs='+', default=[128], help='Optical field size (can be multiple values separated by spaces)')
    parser.add_argument('--sub_optical_field_sizes', type=int, nargs='+', default=[], help='Sub-optical field size (can be multiple values separated by spaces)')
    parser.add_argument('--window_size', type=int, nargs='+', default=None, help='Window size for Extended Hadamard transform')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--imgsz', type=int, default=None, help='Process only a specific image size')
    parser.add_argument('--inverse', action='store_true', help='Order the sub-Hadamard matrices')
    parser.add_argument('--save_aliasing', action='store_true', help='Save the aliasing effect of the Hadamard transform')

    # Parse the command-line arguments
    args = parser.parse_args()
    
    return args

def validate_sizes(sizes):
    # Check if the sizes are powers of two
    for size in sizes:
        if not is_power_of_two(size):
            raise ValueError('Size must be a power of 2')
        
def load_and_preprocess_image(datasetloader, image_path):
    # Load the image
    image_name = os.path.basename(image_path)
    image = datasetloader.load_image(image_path)
    # Get the original dimensions of the image
    original_height, original_width, _ = image.shape
    # Preprocess the image
    image = datasetloader.preprocess(image)
    # Convert the image to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    return image, original_height, original_width, image_name

def process_image(datasetloader, 
                  hadamard_transform, 
                  image_path, 
                  block_size, 
                  sub_optical_field_sizes, 
                  inverse):

    # Load and preprocess the image
    image, original_height, original_width, image_name = load_and_preprocess_image(datasetloader, image_path)

    # Convert the image to optical blocks
    optical_blocks = image_to_optical_blocks(image, block_size)

    # Compute the sum of the optical blocks
    image_sum = np.sum(optical_blocks, axis=(0, 1))

    # Normalize and postprocess the sum image
    save_image_sum = datasetloader.normalize(image_sum)
    save_image_sum = datasetloader.postprocess(save_image_sum, (original_width, original_height))

    # Build the save path for the sum image
    sum_image_path = datasetloader.build_dataset_path(image_name, original=False)

    # Save the sum image
    datasetloader.save(save_image_sum, sum_image_path)

    # Set the sub-optical field size
    datasetloader.update_attributes(sub_optical_field_size=None, inverse=False)

    # Reshape the image sum for Hadamard transform
    channels = image.shape[-1]
    image_reshaped = image_sum.reshape(-1, channels)

    # Apply the Hadamard transform
    hadamard_result = hadamard_transform.transform(image_reshaped)
    if hadamard_transform.optical_field_size in sub_optical_field_sizes:
        # Reshape the result back to the original dimensions
        saved_hadamard_result = hadamard_result.reshape(hadamard_transform.optical_field_size, 
                                                        hadamard_transform.optical_field_size, 
                                                        channels)
        # Normalize and postprocess the Hadamard result
        saved_hadamard_result = datasetloader.normalize(saved_hadamard_result)
        saved_hadamard_result = datasetloader.postprocess(saved_hadamard_result, (original_width, original_height))
        # Build the save path for the Hadamard result
        hadamard_image_path = datasetloader.build_dataset_path(image_name, original=False)
        # Save the Hadamard result
        datasetloader.save(saved_hadamard_result, hadamard_image_path)

    # Iterate over each sub-optical field size
    for sub_optical_field_size in sub_optical_field_sizes:
        # Skip if the sub-optical field size is larger than the optical field size
        if sub_optical_field_size >= hadamard_transform.optical_field_size:
            continue
        # Update the dataset loader with the current sub-optical field size
        datasetloader.update_attributes(sub_optical_field_size=sub_optical_field_size, inverse=inverse)
        # Extract the submatrix from the Hadamard result
        sub_hadamard_result = hadamard_transform.extract_submatrix(hadamard_result, 
                                                                   sub_optical_field_size,
                                                                   inverse=inverse)
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
    inverse = args.inverse

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
                                  imgsz=imgsz,
                                  inverse=inverse)

    # Choose the appropriate Hadamard transform class based on the window size
    hadamard_transform = HadamardTransformExtended() if args.window_size else HadamardTransform()
    if args.seed is not None:
        hadamard_transform.hadmard_matrix_random(args.seed)
    
    # Iterate over each optical field size
    for optical_field_size in optical_field_sizes:
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
                # Reset the sub-optical field size
                datasetloader.update_attributes(sub_optical_field_size=None, inverse=False)
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
                # Normalize and postprocess the sum image
                if args.save_aliasing:
                    save_image_sum = datasetloader.normalize(image_sum)
                    save_image_sum = datasetloader.postprocess(save_image_sum, (original_width, original_height))
                    # Build the save path for the sum image
                    sum_image_path = datasetloader.build_dataset_path(image_name, original=False)
                    # Save the sum image
                    datasetloader.save(save_image_sum, sum_image_path)

                # Set the sub-optical field size
                datasetloader.update_attributes(sub_optical_field_size=optical_field_size)
                # Reshape the image sum for Hadamard transform
                channels = image.shape[-1]
                image_reshaped = image_sum.reshape(-1, channels)
                # Apply the Hadamard transform
                hadamard_result = hadamard_transform.transform(image_reshaped)

                # Iterate over each sub-optical field size
                for sub_optical_field_size in sub_optical_field_sizes:
                    # Update the dataset loader with the current sub-optical field size
                    datasetloader.update_attributes(sub_optical_field_size=sub_optical_field_size, 
                                                    inverse=inverse)
                    # Extract the submatrix from the Hadamard result
                    if sub_optical_field_size > optical_field_size:
                        continue
                    elif sub_optical_field_size == optical_field_size:
                        sub_hadamard_result = hadamard_result
                    else:
                        sub_hadamard_result = hadamard_transform.extract_submatrix(hadamard_result, 
                                                                                   sub_optical_field_size,
                                                                                   inverse=inverse)
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
import cv2
from spi import image_to_optical_blocks, HadamardTransform, HadamardTransformExtended
import numpy as np


if __name__ == '__main__':
    

    # img = cv2.imread('datasets/mouse/images/train/00000.png')

    # block_sizes = (256, 256)

    # print(img.shape)
    
    # image_reshaped = image_to_optical_blocks(img, block_sizes)

    # print(image_reshaped.shape)

    # optical_field_size = 128

    # hadamard_transform1 = HadamardTransform(optical_field_size)

    # print(hadamard_transform1.hadamard_matrix.shape)

    # hadamard_transform2 = HadamardTransformExtended(optical_field_size)

    # print(hadamard_transform2.hadamard_matrix.shape)

    import numpy as np

    # 定义N的值
    N = 5
    # 设置随机数种子
    seed = 123
    np.random.seed(seed)
    # 生成从1到N的随机排列
    random_sequence = np.random.permutation(N)

    # 假设我们有一个5x3的矩阵
    matrix = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    [10, 11, 12],
                    [13, 14, 15]])

    print("原始矩阵:")
    print(matrix)

    # 根据随机序列重新排列矩阵的行
    # 注意：random_sequence 是从1开始的，而Python的索引是从0开始的，所以需要减1
    reordered_matrix = matrix[random_sequence]

    print("重新排列后的矩阵:")
    print(reordered_matrix)






# This code is used to transform the matrix in Hadamard.

from scipy.linalg import hadamard
from .utils import is_power_of_two
import numpy as np
import torch

class HadamardTransform:
    def __init__(self, optical_field_size=8, is_torch=False):
        self.is_torch = is_torch
        self.optical_field_size = optical_field_size
        
    @property
    def optical_field_size(self):
        return self._optical_field_size
    
    @optical_field_size.setter
    def optical_field_size(self, value):
        if not is_power_of_two(value):
            raise ValueError("Optical field size must be a power of 2.")
        self._optical_field_size = value
        self.update_parameters()
    
    def update_parameters(self):
        self.matrix_size = self._optical_field_size ** 2
        if self.is_torch:
            self.hadamard_matrix = torch.tensor(hadamard(self.matrix_size))
        else:
            self.hadamard_matrix = hadamard(self.matrix_size)
        self.hadamard_inverse_matrix = self.hadamard_matrix.T / self.matrix_size
    
    def _check_matrix(self, matrix):
        if matrix.ndim > 2:
            raise ValueError("We only support 1D or 2D matrix.")
        return matrix if matrix.ndim == 2 else matrix[:, np.newaxis]
    
    def transform(self, matrix):
        if self.is_torch and not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)
        matrix = self._check_matrix(matrix)
        return self.hadamard_matrix @ matrix

    def inverse_transform(self, matrix):
        matrix = self._check_matrix(matrix)
        if self.is_torch and not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)
        return self.hadamard_inverse_matrix @ matrix
    
    def extract_submatrix(self, hadamard_result, sub_optical_field_size, order=False):
        if not is_power_of_two(sub_optical_field_size):
            raise ValueError("Sub optical field size must be a power of 2.")
        if sub_optical_field_size >= self._optical_field_size:
            raise ValueError("Sub optical field size must be smaller than the optical field size.")
        sub_matrix_size = sub_optical_field_size ** 2
        if not order:
            return hadamard_result[:sub_matrix_size, :]
        else:
            return hadamard_result[-sub_matrix_size:, :][::-1]
    
    
    def hadmard_matrix_random(self, seed=20241021):
        # create a random seed
        np.random.seed(seed)
        # create a random hadamard matrix
        sorted = np.random.permutation(self.matrix_size)
        self.hadamard_matrix = self.hadamard_matrix[sorted]
        self.hadamard_inverse_matrix = self.hadamard_inverse_matrix[sorted]
            

class HadamardTransformExtended(HadamardTransform):
    def __init__(self, optical_field_size=8, window_size=(2, 2), is_torch=False):
        super().__init__(optical_field_size=optical_field_size, is_torch=is_torch)
        self.window_size = window_size
    
    @property
    def window_size(self):
        return self._window_size
    
    @window_size.setter
    def window_size(self, value):
        self._window_size = value
        self.window = np.ones(value)
        self.window_ones_size = value[0] * value[1]
        self.update_parameters()
        self.create_extended_hadamard_matrix()
    
    def create_extended_hadamard_matrix(self):
        base_matrix = self.hadamard_matrix.reshape(self.matrix_size, self._optical_field_size, self._optical_field_size)
        if self.is_torch:
            self.window = torch.tensor(self.window)
            extended_matrix = torch.kron(base_matrix, self.window)
        else:
            extended_matrix = np.kron(base_matrix, self.window)
        self.hadamard_matrix = extended_matrix.reshape(self.matrix_size, -1)
        self.hadamard_inverse_matrix = self.hadamard_matrix.T / (self.window_ones_size * self.matrix_size)
        
        
        
    
    



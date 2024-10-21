import cv2
import numpy as np
from .hadamard_transform import HadamardTransform, HadamardTransformExtended

__Transform__ = {
    'hadamard': HadamardTransform,
    'hadamard_extended': HadamardTransformExtended
}

class Encoder:
    def __init__(self, encode_method='hadamard', **kwargs):
        self._encode_method = None
        self._transform_kwargs = kwargs  # 保存额外参数
        self.encode_method = encode_method

    @property
    def encode_method(self):
        return self._encode_method

    @encode_method.setter
    def encode_method(self, encode_method):
        if encode_method not in __Transform__:
            raise ValueError('Invalid encode method: {}'.format(encode_method))
        
        self._encode_method = encode_method
        self._initialize_transform()

    def _initialize_transform(self):
        if self._encode_method:
            self._transform = __Transform__[self._encode_method](**self._transform_kwargs)

    def set_transform_kwargs(self, **kwargs):
        self._transform_kwargs = kwargs
        self._initialize_transform()

    def encode(self, img):
        return self._transform.transform(img)
    
    def decode(self, img):
        return self._transform.inverse_transform(img)
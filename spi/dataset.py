from typing import Optional, Tuple
import os
import cv2
import numpy as np

__all__ = ['SPIDataloader', 'SPIVideoLoader']

class SPIDataloader:
    def __init__(self, dataset_input_base_path: str = 'datasets/mouse', 
                 split: str = 'train', 
                 optical_field_size: Optional[int] = None, 
                 sub_optical_field_size: Optional[int] = None, 
                 window_size: Optional[Tuple[int, int]] = None, 
                 hadamard_seed: Optional[int] = None, 
                 inverse: bool = False, 
                 imgsz: Optional[int] = None,
                 aliasing: bool = False):
        
        self.dataset_input_base_path = dataset_input_base_path
        self.split = split
        self.optical_field_size = optical_field_size
        self.sub_optical_field_size = sub_optical_field_size
        self.window_size = window_size
        self.hadamard_seed = hadamard_seed
        self.inverse = inverse
        self.aliasing = aliasing
        self.imgsz = imgsz
    
    def update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found in {self.__class__.__name__}")

    def build_dataset_path(self, image_name: str = "", original: bool = True) -> str:
        path_parts = [self.dataset_input_base_path]
        
        if not original:
            part = []
            if self.optical_field_size is not None:
                part.append(f'images-{self.optical_field_size}x{self.optical_field_size}')
            if self.sub_optical_field_size is not None:
                part.append(f'{self.sub_optical_field_size}x{self.sub_optical_field_size}')
            if self.window_size is not None:
                part.append(f'{self.window_size[0]}x{self.window_size[1]}')
            if self.inverse:
                part.append('inverse')
            if self.aliasing:
                part.append('aliasing')
            if self.imgsz is not None:
                part.append(f'{self.imgsz}')
            if self.hadamard_seed is not None:
                part.append(f'{self.hadamard_seed}')
            path_parts.append('-'.join(part))
        else:
            path_parts.append('images')
        
        path_parts.append(self.split)
        save_path = os.path.join(*path_parts, image_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path

    def load(self, image_name: Optional[str] = None, 
                  batch_size: Optional[int] = None, 
                  original: bool = True, 
                  load_paths: bool = True, 
                  load_images: bool = False):
        split_path = self.build_dataset_path(original=original)
        image_files = [os.path.join(split_path, f) for f in os.listdir(split_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_name is not None:
            image_files = [os.path.join(split_path, image_name)]

        if batch_size is not None:
            image_files = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]

        if load_images and load_paths:
            if batch_size is not None:
                data = [[(image_path, self.load_image(image_path)) for image_path in batch] for batch in image_files]
            else:
                data = [(image_path, self.load_image(image_path)) for image_path in image_files]
        elif load_images:
            if batch_size is not None:
                data = [[self.load_image(image_path) for image_path in batch] for batch in image_files]
            else:
                data = [self.load_image(image_path) for image_path in image_files]
        elif load_paths:
            data = image_files
        else:
            raise ValueError("At least one of load_paths or load_images must be True")

        return data

    def load_image(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_path} not found")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image {image_path}")
        return img
    
    def save(self, 
             img: np.ndarray, 
             image_path: Optional[str] = None):
        if image_path is None:
            raise ValueError("Image path must be provided")
        cv2.imwrite(image_path, img)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.imgsz is not None:
            img = cv2.resize(img, (self.imgsz, self.imgsz))
        return img

    def postprocess(self, img: np.ndarray, imgsz: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if self.imgsz is not None and imgsz is not None:
            img = cv2.resize(img, imgsz)
        return img
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return img
    

class SPIVideodataset:
    def __init__(self, 
                 video_base_path: str = 'videos',
                 optical_field_size: Optional[int] = None, 
                 sub_optical_field_size: Optional[int] = None, 
                 window_size: Optional[Tuple[int, int]] = None, 
                 hadamard_seed: Optional[int] = None, 
                 inverse: bool = False, 
                 imgsz: Optional[int] = None,
                 aliasing: bool = False):
        self.video_base_path = video_base_path
        self.optical_field_size = optical_field_size
        self.sub_optical_field_size = sub_optical_field_size
        self.window_size = window_size
        self.seed = hadamard_seed
        self.inverse = inverse
        self.imgsz = imgsz
        self.aliasing = aliasing
    
    def update_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} not found in {self.__class__.__name__}")
    
    def build_video_path(self, video_name: str = "", original: bool = True) -> str:
        path_parts = [self.video_base_path]
        video_name_ = video_name.split('.')[0]
        video_name_ext = video_name.split('.')[-1]
        
        if not original:
            part = []
            if self.optical_field_size is not None:
                part.append(f'{video_name_}-{self.optical_field_size}x{self.optical_field_size}')
            if self.sub_optical_field_size is not None:
                part.append(f'{self.sub_optical_field_size}x{self.sub_optical_field_size}')
            if self.window_size is not None:
                part.append(f'{self.window_size[0]}x{self.window_size[1]}')
            if self.inverse:
                part.append('inverse')
            if self.aliasing:
                part.append('aliasing')
            if self.imgsz is not None:
                part.append(f'{self.imgsz}')
            if self.seed is not None:
                part.append(f'{self.seed}')
            path_parts.append('-'.join(part))
        else:
            path_parts.append(video_name)
        save_path = os.path.join(*path_parts)
        save_path = save_path + '.' + video_name_ext
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return save_path
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        if self.imgsz is not None:
            img = cv2.resize(img, (self.imgsz, self.imgsz))
        return img

    def postprocess(self, img: np.ndarray, imgsz: Optional[Tuple[int, int]] = None) -> np.ndarray:
        if self.imgsz is not None and imgsz is not None:
            img = cv2.resize(img, imgsz)
        return img
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return img
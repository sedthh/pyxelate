import numpy as np
from skimage.morphology import square as skimage_square
from skimage.morphology import binary_dilation as skimage_dilation


class Vid:
    """Generator class that yields new images based on differences between them"""
    
    def __init__(self, images, pad=0, sobel=2, keyframe=.3, sensitivity=.1):
        assert isinstance(images, (list, tuple)), "Function only accepts list or tuple of image representations!"
        self.images = images
        if pad is None or pad == 0:
            self.pad = (None, None)
        elif isinstance(pad, int):
            self.pad = (pad, -pad)
        elif isinstance(pad, (list, tuple)):
            self.pad = (None if pad[0] == 0 else pad[0], None if pad[1] == 0 else -pad[1])
        else:
            raise ValueError("The value of 'pad' must be int or (int, int)")
        self.sobel = int(sobel)
        self.keyframe = keyframe
        self.sensitivity = sensitivity
    
    def __iter__(self):
        for i, image in enumerate(self.images):
            current_image = np.clip(np.copy(image[self.pad[0]:self.pad[1], :, :3]) / 255., 0., 1.)
            if i == 0:
                last_image = np.copy(current_image)
                key_image = np.copy(current_image)
                yield last_image, True 
            else:
                assert np.all([a == b for a, b in zip(last_image.shape, current_image.shape)]), f"Image at position {i} has different size!"
                last_difference = np.abs(current_image[:, :, :3] - last_image[:, :, :3])
                last_difference = np.max(last_difference, axis=2)
                key_difference = np.abs(current_image[:, :, :3] - key_image[:, :, :3])
                key_difference = np.max(key_difference, axis=2)
                if np.mean(last_difference) < self.keyframe or np.mean(key_difference) < self.keyframe:
                    mask = np.where(key_difference > self.sensitivity, True, False)
                    if self.sobel:
                        mask = skimage_dilation(mask, footprint=skimage_square(self.sobel * 6))
                    mask = np.expand_dims(mask, axis=-1)
                    last_image = current_image * mask + last_image * (1. - mask)
                    yield last_image, False    
                else:
                    # new keyframe
                    last_image = np.copy(current_image)
                    key_image = np.copy(current_image)
                    yield last_image, True   

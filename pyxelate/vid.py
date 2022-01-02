import numpy as np
from sklearn.utils.extmath import randomized_svd
from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.transform import resize
from skimage.morphology import square as skimage_square
from skimage.morphology import dilation as skimage_dilation

class images_to_parts:
    
    SVD_N_COMPONENTS = 32
    SVD_MAX_ITER = 16
    SVD_RANDOM_STATE = 1234
    
    def __init__(self, images, square=2, keyframe=.66, sensitivity=.1):
        assert isinstance(images, (list, tuple)), "Function only accepts list or tuple of image representations!"
        self.images = images
        self.square = int(square)
        self.keyframe = keyframe
        self.sensitivity = sensitivity
    
    def __iter__(self):
        for i, image in enumerate(self.images):
            if not self._is_transparent(image):
                current_image = np.dstack((np.copy(image), np.ones((image.shape[0], image.shape[1])) * 255))
            else:
                current_image = np.copy(image)
            current_image = np.clip(current_image / 255., 0., 1.)
            if i == 0:
                yield current_image, True
            else:
                current_svd = self._svd(current_image)
                assert np.all([a == b for a, b in zip(last_image.shape, current_image.shape)]), f"Image at position {i} has different size!"
                difference = np.abs(current_svd[:, :, :3] - last_svd[:, :, :3])
                mask = np.where(np.max(difference, axis=2) > self.sensitivity, True, False)
                if self.square:
                    mask = skimage_dilation(mask, footprint=skimage_square(self.square))
                if np.sum(mask) < self.keyframe * np.prod(mask.shape):
                    current_image[:, :, 3] = mask
                    yield current_image, False
                else:
                    # new keyframe
                    yield current_image, True
            last_image = np.copy(current_image)
            last_svd = self._svd(current_image)
    
    
    @staticmethod
    def _is_transparent(X):
        """Returns True if image has a dimension for transparency"""
        return bool(X.shape[2] == 4)
    
    def _svd(self, X):
        """Reconstruct image via truncated SVD on each RGB channel"""
        if self.SVD_N_COMPONENTS >= X.shape[0] and self.SVD_N_COMPONENTS >= X.shape[1]:
            return X  # skip SVD
                
        @adapt_rgb(each_channel)
        def _wrapper(dim):    
            U, s, V = randomized_svd(dim, 
                                    n_components=self.SVD_N_COMPONENTS,
                                    n_iter=self.SVD_MAX_ITER,
                                    random_state=self.SVD_RANDOM_STATE)
            S = np.diag(s.ravel())
            return U.dot(S.dot(V))  # NOTE: no casting to 0.-1.
            
        return _wrapper(X)


class parts_to_images:
 
    def __init__(self, images, keyframes=[]):
        assert isinstance(images, (list, tuple)), "Function only accepts list or tuple of image representations!"
        self.images = images
        if not keyframes:
            self.keyframes = [None for _ in range(len(self.images))]
        else:
            assert isinstance(keyframes, (list, tuple)), "Function only accepts list or tuple of booleans!"
            self.keyframes = keyframes
        
    def __iter__(self):
        for i, (image, is_keyframe) in enumerate(zip(self.images, self.keyframes)):
            if i == 0:
                last_image = np.copy(image)
            else:
                if is_keyframe:
                    last_image = np.copy(image)
                else:
                    last_image = self.merge_images(last_image, image)
            yield last_image
                                  
    @staticmethod      
    def merge_images(*args):
        assert len(args), "No images given!"
        result = np.copy(args[0])
        for image in args[1:]:
            mask = ~np.logical_xor(result[:, :, 3], image[:, :, 3])
            result[mask] = image[mask]
        return result
    
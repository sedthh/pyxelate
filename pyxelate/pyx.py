import numpy as np
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import BayesianGaussianMixture
from sklearn.utils.extmath import randomized_svd

from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, deltaE_ciede2000
from skimage.exposure import equalize_adapthist
from skimage.filters import sobel as skimage_sobel
from skimage.filters import median as skimage_median
from skimage.morphology import square as skimage_square
from skimage.morphology import dilation as skimage_dilation
from skimage.transform import resize
from skimage.util import view_as_blocks

from scipy.ndimage import convolve

from numba import njit

try:
    from .pal import BasePalette
except ImportError:
    from pal import BasePalette

from typing import Optional, Union, Tuple


class PyxWarning(Warning):
    def __init__(self, message):
        self.message = message
        
    def __str__(self):
        return repr(self.message)


class BGM(BayesianGaussianMixture):
    """Wrapper for BayesianGaussianMixture"""
    MAX_ITER = 128
    RANDOM_STATE = 1234567
    
    def __init__(self, palette: Union[int, BasePalette], find_palette: bool) -> None:
        """Init BGM with different default parameters depending on use-case"""
        self.palette = palette
        self.find_palette = find_palette
        if self.find_palette:
            super().__init__(n_components=self.palette,
                    max_iter=self.MAX_ITER,
                    covariance_type="tied",
                    weight_concentration_prior_type="dirichlet_distribution",  
                    weight_concentration_prior=1. / self.palette, 
                    mean_precision_prior=1. / 256.,
                    warm_start=False,
                    random_state=self.RANDOM_STATE)
        else:
            super().__init__(n_components=len(self.palette),
                    max_iter=self.MAX_ITER,
                    covariance_type="tied",
                    weight_concentration_prior_type="dirichlet_process",  
                    weight_concentration_prior=1e-7,
                    mean_precision_prior=1. / len(self.palette),
                    warm_start=False,
                    random_state=self.RANDOM_STATE)
            # start centroid search from the palette's values
            self.mean_prior = np.mean([val[0] for val in self.palette], axis=0)
    
    def _initialize_parameters(self, X: np.ndarray, random_state: int) -> None:
        """Changes init parameters from K-means to CIE LAB distance when palette is assigned"""
        assert self.init_params == "kmeans", "Initialization is overwritten, can only be set as 'kmeans'."
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        if self.find_palette:
            # original centroids
            label = KMeans(n_clusters=self.n_components, n_init=1,
                                   random_state=random_state).fit(X).labels_
        else:
            # color distance based centroids
            label = np.argmin([deltaE_ciede2000(rgb2lab(X), rgb2lab(p), kH=3, kL=2) for p in self.palette], axis=0)
        resp[np.arange(n_samples), label] = 1
        self._initialize(X, resp)
        
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BGM":
        """Fits BGM model but alters convergence warning"""
        converged = True
        with warnings.catch_warnings(record=True) as w:
            super().fit(X)
            if w and w[-1].category == ConvergenceWarning:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                converged = False
        if not converged:
            warnings.warn("Pyxelate could not properly assign colors, try a different palette size for better results!", PyxWarning)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = super().predict_proba(X)
        # adjust for monochrome
        if self.find_palette:
            if self.palette < 3:
                return np.sqrt(p)
        elif len(self.palette) < 3:
            return np.sqrt(p)
        return p


class Pyx(BaseEstimator, TransformerMixin):
    """Pyx extends scikit-learn transformers"""
    
    BGM_RESIZE = 256
    SCALE_RGB = 1.07
    HIST_BRIGHTNESS = 1.19
    COLOR_QUANT = 8
    DITHER_AUTO_SIZE_LIMIT_HI = 512
    DITHER_AUTO_SIZE_LIMIT_LO = 16
    DITHER_AUTO_COLOR_LIMIT = 8
    DITHER_NAIVE_BOOST = 1.33
    SVD_N_COMPONENTS = 32
    SVD_MAX_ITER = 16
    SVD_RANDOM_STATE = 1234
    # precalculated 4x4 Bayer Matrix / 16 - 0.5
    DITHER_BAYER_MATRIX = np.array([[-0.5   ,  0.    , -0.375 ,  0.125 ],
                                   [ 0.25  , -0.25  ,  0.375 , -0.125 ],
                                   [-0.3125,  0.1875, -0.4375,  0.0625],
                                   [ 0.4375, -0.0625,  0.3125, -0.1875]])
    
    def __init__(self, 
                 height: Optional[int] = None, 
                 width: Optional[int] = None, 
                 factor: Optional[int] = None, 
                 upscale: Union[Tuple[int, int], int] = 1, 
                 depth: int = 1, 
                 palette: Union[int, BasePalette] = 8, 
                 dither: Optional[str] = "none", 
                 sobel: int = 3, 
                 svd: bool = True,
                 alpha: float = .6) -> None:
        if (width is not None or height is not None) and factor is not None:
            raise ValueError("You can only set either height + width or the downscaling factor, but not both!")
        assert height is None or height >= 1, "Height must be a positive integer!"
        assert width is None or width >= 1, "Width must be a positive integer!" 
        assert factor is None or factor >= 1, "Factor must be a positive integer!"
        assert isinstance(sobel, int) and sobel >= 2, "Sobel must be an integer strictly greater than 1!"
        self.height = int(height) if height else None
        self.width = int(width) if width else None
        self.factor = int(factor) if factor else None
        self.sobel = sobel
        if isinstance(upscale, (list, tuple, set, np.ndarray)):
            assert len(upscale) == 2, "Upscale must be len 2, with 2 positive integers!"
            assert upscale[0] >= 1 and upscale[1] >=1, "Upscale must have 2 positive values!"
            self.upscale = (upscale[0], upscale[1])
        else:    
            assert upscale >= 1, "Upscale must be a positive integer!"
            self.upscale = (upscale, upscale)
        assert depth > 0 and isinstance(depth, int), "Depth must be a positive integer!"
        if depth > 2:
            warnings.warn("Depth too high, it will probably take really long to finish!", PyxWarning)
        self.depth = depth
        self.palette = palette
        self.find_palette = isinstance(self.palette, (int, float))  # palette is a number
        if self.find_palette and palette < 2:
            raise ValueError("The minimum number of colors in a palette is 2")
        elif not self.find_palette and len(palette) < 2:
            raise ValueError("The minimum number of colors in a palette is 2")
        assert dither in (None, "none", "naive", "bayer", "floyd", "atkinson"), "Unknown dithering algorithm!"
        self.dither = dither
        self.svd = bool(svd)
        self.alpha = float(alpha)
        # instantiate BGM model
        self.model = BGM(self.palette, self.find_palette)
        self.is_fitted = False
        self.palette_cache = None
    
    def _get_size(self, original_height: int, original_width: int) -> Tuple[int, int]:
        """Calculate new size depending on settings"""
        if self.height is not None and self.width is not None:
            return self.height, self.width
        elif self.height is not None:
            return self.height, int(self.height / original_height * original_width)
        elif self.width is not None:
            return int(self.width / original_width * original_height), self.width
        elif self.factor is not None:
            return original_height // self.factor, original_width // self.factor
        else:
            return original_height, original_width

    def _image_to_float(self, image: np.ndarray) -> np.ndarray:
        """Helper function that changes 0 - 255 color representation to 0. - 1."""
        if np.issubdtype(image.dtype, np.integer):
            return np.clip(np.array(image, dtype=float) / 255., 0, 1).astype(float)
        return image
    
    def _image_to_int(self, image: np.ndarray) -> np.ndarray:
        """Helper function that changes 0. - 1. color representation to 0 - 255"""
        if isinstance(image, BasePalette):
            image = np.array(image.value, dtype=float)
        elif isinstance(image, (list, tuple)):
            is_int = np.all([isinstance(x, int) for x in image])
            if is_int:
                return np.clip(np.array(image, dtype=int), 0, 255)
            else:
                image = np.array(image, dtype=float)
        if image.dtype in (float, np.float32, np.float64):  # np.float is deprecated
            return np.clip(np.array(image, dtype=float) * 255., 0, 255).astype(int)
        return image
        
    @property
    def colors(self) -> np.ndarray:
        """Get colors in palette (0 - 255 range)"""
        if self.palette_cache is None:
            if self.find_palette:
                assert self.is_fitted, "Call 'fit(image_as_numpy)' first!"
                c = rgb2hsv(self.model.means_.reshape(-1, 1, 3))
                c[:, :, 1:] *= self.SCALE_RGB
                c = hsv2rgb(c)
                c = np.clip(c * 255 // self.COLOR_QUANT * self.COLOR_QUANT, 0, 255).astype(int)
                c[c < self.COLOR_QUANT * 2] = 0
                c[c > 255 - self.COLOR_QUANT * 2] = 255
                self.palette_cache = c
                if len(np.unique([f"{pc[0]}" for pc in self.palette_cache])) != len(c):
                    warnings.warn("Some colors are redundant, try a different palette size for better results!", PyxWarning)
            else:
                self.palette_cache = self._image_to_int(self.palette)
        return self.palette_cache
    
    @property
    def _palette(self) -> np.ndarray:
        """Get colors in palette as a plottable palette format (0. - 1. in correct shape)"""
        return self._image_to_float(self.colors.reshape(-1, 3))
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Pyx":
        """Fit palette and optionally calculate automatic dithering"""
        h, w, d = X.shape
        # create a smaller image for BGM without alpha channel
        if d > 3:
            # separate color and alpha channels
            X_ = self._dilate(X).reshape(-1, 4)
            alpha_mask = X_[:, 3]
            X_ = X_[alpha_mask >= self.alpha]
            X_ = X_.reshape(1, -1, 4)
            X_ = resize(X[:, :, :3], (1, min(h, self.BGM_RESIZE) * min(w, self.BGM_RESIZE)), anti_aliasing=False)
        else:
            X_ = resize(X[:, :, :3], (min(h, self.BGM_RESIZE), min(w, self.BGM_RESIZE)), anti_aliasing=False)
        X_ = self._image_to_float(X_).reshape(-1, 3)  # make sure colors have a float representation
        if self.find_palette:
            X_ = ((X_ - .5) * self.SCALE_RGB) + .5  # move values away from grayish colors 
        
        # fit BGM to generate palette
        self.model.fit(X_)
        self.is_fitted = True  # all done, user may call transform()
        return self

    def _pyxelate(self, X: np.ndarray) -> np.ndarray:
        """Downsample image based on the magnitude of its gradients in sobel-sided tiles"""

        @adapt_rgb(each_channel)
        def _wrapper(channel):
            # apply to all RGB channels
            sobel = skimage_sobel(channel)
            sobel += 1e-8 # avoid division by zero
            sobel_norm = view_as_blocks(sobel, (self.sobel, self.sobel)).sum((2,3))
            sum_prod = view_as_blocks((sobel * channel), (self.sobel, self.sobel)).sum((2,3))
            return sum_prod / sobel_norm

        X_pad = self._pad(X, self.sobel)
        return _wrapper(X_pad).copy()
    
    def _pad(self, X: np.ndarray, 
             pad_size: int, 
             nh: Optional[int] = None, 
             nw: Optional[int] = None) -> np.ndarray:
        """Pad image if it's not pad_size divisable or remove such padding"""
        if nh is None and nw is None:
            # pad edges so image is divisible by pad_size
            h, w, d = X.shape
            h1, h2 = (1 if h % pad_size > 0 else 0), (1 if h % pad_size == 1 else 0)
            w1, w2 = (1 if w % pad_size > 0 else 0), (1 if w % pad_size == 1 else 0)
            return np.pad(X, ((h1, h2), (w1, w2), (0, 0)), "edge")
        else:
            # remove previous padding
            return X[slice((1 if nh % pad_size > 0 else 0),(-1 if nh % pad_size == 1 else None)), 
                     slice((1 if nw % pad_size > 0 else 0),(-1 if nw % pad_size == 1 else None)), :]
    
    def _dilate(self, X: np.ndarray) -> np.ndarray:
        """Dilate semi-transparent edges to remove artifacts (for images with opacity)"""
        
        @adapt_rgb(each_channel)
        def _wrapper(channel):
            # apply to each channel
            return skimage_dilation(channel, footprint=skimage_square(3))
        
        h, w, d = X.shape
        X_ = self._pad(X, 3)
        mask = X_[:, :, 3]
        alter = _wrapper(X_[:, :, :3])
        X_[:, :, :3][mask < self.alpha] = alter[mask < self.alpha]
        return self._pad(X_, 3, h, w)
    
    def _median(self, X: np.ndarray) -> np.ndarray:
        """Custom median filter on HSV channels using 3x3 squares"""
        
        @adapt_rgb(each_channel)
        def _wrapper(channel):
            # apply to each channel
            return skimage_median(channel, skimage_square(3))
        
        h, w, d = X.shape
        X_ = self._pad(X, 3)  # add padding for median filter
        X_ = rgb2hsv(X_)  # change to HSV
        X_ = _wrapper(X_)  # apply median filter
        X_ = hsv2rgb(X_)  # go back to RGB
        return self._pad(X_, 3, h, w)  # remove added padding

    def _warn_on_dither_with_alpha(self, d: int) -> None:
        if d > 3 and self.dither in ("bayer", "floyd", "atkinson"):
            warnings.warn("Images with transparency can have unwanted artifacts around the edges with this dithering method. Use 'naive' instead.", PyxWarning)

    def _svd(self, X):
        """Reconstruct image via truncated SVD on each RGB channel"""
        if self.SVD_N_COMPONENTS >= X.shape[0] - 1 and self.SVD_N_COMPONENTS >= X.shape[1] - 1:
            return X  # skip SVD
                
        @adapt_rgb(each_channel)
        def _wrapper(dim):    
            U, s, V = randomized_svd(dim, 
                                    n_components=self.SVD_N_COMPONENTS,
                                    n_iter=self.SVD_MAX_ITER,
                                    random_state=self.SVD_RANDOM_STATE)
            S = np.diag(s.ravel())
            A = U.dot(S.dot(V))
            return np.clip(A / 255., 0., 1.)
        
        return _wrapper(X)

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform image to pyxelated version"""
        assert self.is_fitted, "Call 'fit(image_as_numpy)' first before calling 'transform(image_as_numpy)'!"
        h, w, d = X.shape
        if self.find_palette:
            assert h * w > self.palette, "Too many colors for such a small image! Use a larger image or a smaller palette."
        else:
            assert h * w > len(self.palette), "Too many colors for such a small image! Use a larger image or a smaller palette."
        
        new_h, new_w = self._get_size(h, w)  # get desired size depending on settings
        if d > 3:
            # image has alpha channel
            X_ = self._dilate(X)
            alpha_mask = resize(X_[:, :, 3], (new_h, new_w), anti_aliasing=True)
        else:
            # image has no alpha channel
            X_ = X
            alpha_mask = None
        if self.depth:
            # change size depending on the number of iterations
            new_h, new_w = new_h * (self.sobel ** self.depth), new_w * (self.sobel ** self.depth)
        X_ = resize(X_[:, :, :3], (new_h, new_w), anti_aliasing=True)  # colors are now 0. - 1.        
        
        # optionally apply svd for a somewhat blockier low-pass look
        if self.svd:
            X_ = self._svd(X_)
        
        # adjust contrast
        X_ = rgb2hsv(equalize_adapthist(X_))  # to hsv after local contrast fix
        X_[:, :, 1:] *= self.HIST_BRIGHTNESS  # adjust v only
        X_ = hsv2rgb(np.clip(X_, 0., 1.))  # back to rgb
            
        # pyxelate iteratively
        for _ in range(self.depth):
            if d == 3:
                # remove noise
                X_ = self._median(X_)
            X_ = self._pyxelate(X_)  # downsample in each iteration
            
        final_h, final_w, _ = X_.shape
        if self.find_palette:
            X_ = ((X_ - .5) * self.SCALE_RGB) + .5  # values were already altered before in .fit()
        reshaped = np.reshape(X_, (final_h * final_w, 3))
            
        # add dithering, took a lot of ideas from https://surma.dev/things/ditherpunk/
        if self.dither is None or self.dither == "none":
            probs = self.model.predict(reshaped)
            X_ = self.colors[probs]
        elif self.dither == "naive":
            # pyxelate dithering based on BGM probability density only
            probs = self.model.predict_proba(reshaped)
            p = np.argmax(probs, axis=1)
            X_ = self.colors[p]
            probs[np.arange(len(p)), p] = 0
            p2 = np.argmax(probs, axis=1)  # second best
            v1 = np.max(probs, axis=1) > (1.  / (len(self.colors) + 1))
            v2 = np.max(probs, axis=1) > (1.  / (len(self.colors) * self.DITHER_NAIVE_BOOST + 1))
            pad = not bool(final_w % 2)
            for i in range(0, len(X_), 2):
                m = (i // final_w) % 2
                if pad:
                    i += m
                if m:
                    if v1[i]:
                        X_[i] = self.colors[p2[i]]
                elif v2[i]:
                    X_[i] = self.colors[p2[i]]
        elif self.dither == "bayer":
            # Bayer-like dithering
            self._warn_on_dither_with_alpha(d)
            probs = self.model.predict_proba(reshaped)
            probs = [convolve(probs[:, i].reshape((final_h, final_w)), self.DITHER_BAYER_MATRIX, mode="reflect") for i in range(len(self.colors))]
            probs = np.argmin(probs, axis=0)
            X_ = self.colors[probs]
        elif self.dither == "floyd":
            # Floyd-Steinberg-like algorithm
            self._warn_on_dither_with_alpha(d)
            X_ = self._dither_floyd(reshaped, (final_h, final_w))
        elif self.dither == "atkinson":
            # Atkinson-like algorithm
            self._warn_on_dither_with_alpha(d)
            res = np.zeros((final_h + 2, final_w + 3), dtype=int)
            X_ = np.pad(X_, ((0, 2), (1, 2), (0, 0)), "reflect")
            for y in range(final_h):
                for x in range(1, final_w+1):
                    pred = self.model.predict_proba(X_[y, x, :3].reshape(-1, 3))
                    res[y, x] = np.argmax(pred)
                    quant_error = (X_[y, x, :3] - self.model.means_[res[y, x]]) / 8.
                    X_[y, x+1, :3] += quant_error
                    X_[y, x+2, :3] += quant_error
                    X_[y+1, x-1, :3] += quant_error
                    X_[y+1, x, :3] += quant_error
                    X_[y+1, x+1, :3] += quant_error
                    X_[y+2, x, :3] += quant_error
            # fix edges
            res = res[:final_h, 1:final_w+1]
            X_ = self.colors[res.reshape(final_h * final_w)]
        
        X_ = np.reshape(X_, (final_h, final_w, 3))  # reshape to actual image dimensions
        if alpha_mask is not None:
            # attach lost alpha layer
            alpha_mask[alpha_mask >= self.alpha] = 255
            alpha_mask[alpha_mask < self.alpha] = 0
            X_ = np.dstack((X_[:, :, :3], alpha_mask.astype(int)))
        
        # return upscaled image
        X_ = np.repeat(np.repeat(X_, self.upscale[0], axis=0), self.upscale[1], axis=1)
        return X_.astype(np.uint8)

    def _dither_floyd(self, reshaped: np.ndarray, final_shape: Tuple[int, int]) -> np.ndarray:
        """Floyd-Steinberg-like dithering (multiple steps are applied for speed up)"""

        @njit()
        def _wrapper(probs, final_h, final_w):
            #probs = 1. / np.where(probs == 1, 1., -np.log(probs))
            probs = np.power(probs, (1. / 6.))
            res = np.zeros((final_h, final_w), dtype=np.int8)
            for y in range(final_h - 1):
                for x in range(1, final_w - 1):
                    quant_error = probs[:, y, x] / 16.
                    res[y, x] = np.argmax(quant_error)
                    quant_error[res[y, x]] = 0.
                    probs[:, y, x+1] += quant_error * 7.
                    probs[:, y+1, x-1] += quant_error * 3.
                    probs[:, y+1, x] += quant_error * 5.
                    probs[:, y+1, x+1] += quant_error
            # fix edges
            x = final_w - 1
            for y in range(final_h):
                res[y, x] = np.argmax(probs[:, y, x])
                res[y, 0] = np.argmax(probs[:, y, 0])
            y = final_h - 1
            for x in range(1, final_w - 1):
                res[y, x] = np.argmax(probs[:, y, x])
            return res
        
        final_h, final_w = final_shape
        probs = self.model.predict_proba(reshaped)
        probs = np.array([probs[:, i].reshape((final_h, final_w)) for i in range(len(self.colors))])
        res = _wrapper(probs, final_h, final_w)
        return self.colors[res.reshape(final_h * final_w)]

    
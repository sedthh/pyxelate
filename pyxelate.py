import numpy as np
import warnings

from skimage.color.adapt_rgb import adapt_rgb, each_channel
from skimage.util import view_as_blocks
from skimage.morphology import square, dilation
from skimage.filters import median
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import equalize_adapthist
from skimage.transform import resize

from sklearn.mixture import BayesianGaussianMixture
from sklearn.exceptions import ConvergenceWarning

__version__ = '1.2.1'
__version_info__ = tuple(int(num) for num in __version__.split('.'))


class Pyxelate:

	CONVOLUTIONS = np.array([
		[[2, 2], [2, 2]],

		[[11, -1], [-1, -1]],
		[[-1, 11], [-1, -1]],
		[[-1, -1], [11, -1]],
		[[-1, -1], [-1, 11]],

		[[5, 5], [-1, -1]],
		[[-1, -1], [5, 5]],

		[[5, -1], [5, -1]],
		[[-1, 5], [-1, 5]],

		[[5, -1], [-1, 5]],
		[[-1, 5], [5, -1]],

		[[-1, 3], [3, 3]],
		[[3, -1], [3, 3]],
		[[3, 3], [-1, 3]],
		[[3, 3], [3, -1]]
	], dtype="int")

	SOLUTIONS = np.array([
		[[1, 1], [1, 1]],

		[[0, 1], [1, 1]],
		[[1, 0], [1, 1]],
		[[1, 1], [0, 1]],
		[[1, 1], [1, 0]],

		[[1, 1], [0, 0]],
		[[0, 0], [1, 1]],

		[[1, 0], [1, 0]],
		[[0, 1], [0, 1]],

		[[1, 0], [1, 0]],
		[[0, 1], [0, 1]],

		[[1, 0], [0, 0]],
		[[0, 1], [0, 0]],
		[[0, 0], [1, 0]],
		[[0, 0], [0, 1]],
	], dtype="bool")

	ITER = 2

	def __init__(self, height, width, color=8, dither=True, alpha=.6, regenerate_palette=True,
				 keyframe=.6, sensitivity=.07, random_state=0):
		"""Create instance for generating similar pixel arts."""
		self.height = int(height)
		self.width = int(width)
		if self.width < 1 or self.height < 1:
			raise ValueError("Result can not be smaller than 1x1 pixels.")
		self.color = int(color)
		if self.color < 2:
			raise ValueError("The minimum number of colors is 2.")
		elif self.color > 32:
			raise ValueError("The maximum number of colors is 32.")
		if dither:
			self.dither = 1 / (self.color + 1)
		else:
			self.dither = 0.
		self.alpha = float(alpha)   # threshold for opacity
		self.regenerate_palette = bool(regenerate_palette)
		self.keyframe = keyframe  # threshold for differences between keyframes
		self.sensitivity = sensitivity  # threshold for differences between parts of keyframes

		# BGM
		self.is_fitted = False
		self.random_state = int(random_state)
		self.model = BayesianGaussianMixture(n_components=self.color,
											 max_iter=256,
											 covariance_type="tied",
											 weight_concentration_prior_type="dirichlet_distribution",
											 mean_precision_prior=1. / 256.,
											 warm_start=False,
											 random_state=self.random_state)

	def convert(self, image):
		"""Generate pixel art from image"""
		return self._convert(image, False, False)

	def _convert(self, image, override_adapthist=False, override_dither=False):
		"""Generate pixel art from image or sequence of images"""
		# does the image have alpha channel?
		if self._is_transparent(image):
			# remove artifacts from transparent edges
			image = self._dilate(image)
			# create alpha mask
			mask = resize(image[:, :, 3], (self.height, self.width), anti_aliasing=True)
			# mask for colors
			color_mask = resize(image[:, :, 3], (32, 32), anti_aliasing=False).ravel()
		else:
			mask = None
			color_mask = None

		# apply adaptive contrast
		if not override_adapthist:
			image = self._fix_hist(image)

		# create sample for finding palette
		if self.regenerate_palette or not self.is_fitted:
			examples = resize(image[:, :, :3], (32, 32), anti_aliasing=False).reshape(-1, 3).astype("int")
			if color_mask is not None:
				# transparent colors should be ignored
				examples = examples[color_mask >= self.alpha]
			self._fit_model(examples)

		# resize image to 4 times the desired width and height
		image = resize(image[:, :, :3], (self.height * self.ITER * 2, self.width * self.ITER * 2), anti_aliasing=True)
		# generate pixelated image with desired width / height
		image = self._reduce(image)

		# apply palette
		height, width, depth = image.shape
		reshaped = np.reshape(image, (height * width, depth))
		probs = self.model.predict_proba(reshaped)
		y = np.argmax(probs, axis=1)

		# increase hue and snap color values to multiples of 8
		palette = rgb2hsv(self.model.means_.reshape(-1, 1, 3))
		palette[:, :, 1] *= 1.14  # empirical magic number
		palette = hsv2rgb(palette).reshape(self.color, 3) // 8 * 8
		palette[palette == 248] = 255  # clamping // 8 * 8 would rarely allow 255 values

		# generate recolored image
		image = palette[y]

		# apply dither over threshold if it's not zero
		if not override_dither and self.dither:
			# get second best probability by removing the best one
			probs[np.arange(len(y)), y] = 0
			# get new best and values
			v = np.max(probs, axis=1) > self.dither
			y = np.argmax(probs, axis=1)

			# replace every second pixel with second best color
			pad = not bool(width % 2)
			if pad:
				# make sure to alternate between starting positions
				# bottleneck
				for i in range(0, len(image), 2):
					i += (i // width) % 2
					if v[i]:
						image[i] = palette[y[i]]
			else:
				i = np.argwhere(v[::2]) * 2
				image[i] = palette[y[i]]

		image = np.reshape(image, (height, width, depth))
		if mask is not None:
			# use transparency from original image, but make it either 0 or 255
			mask[mask >= self.alpha] = 255
			mask[mask < self.alpha] = 0
			image = np.dstack((image, mask))  # result has lost its alpha channel

		return np.clip(image.astype("int"), 0, 255).astype("uint8")

	def convert_sequence(self, images):
		"""Generates sequence of pixel arts from a list of images"""
		try:
			_ = np.array(images, dtype=float)
		except ValueError:
			# image sizes are different == setting an array element with a sequence
			raise ValueError("Shape of images in list are different.")

		# apply adaptive histogram on each
		images = [self._fix_hist(image) for image in images]

		transparent = self._is_transparent(images[0])
		keyframe_limit = self.keyframe * np.prod(images[0].shape) * 255.
		sensitivity_limit = self.sensitivity * 255.
		diff_images, key_frames = [], []

		# create new images that are just the differences between sequences
		for image in images:
			# add first image
			if diff_images:
				diff = np.abs(image[:, :, :3] - diff_images[-1][:, :, :3])
				# image is not too different, from previous one, create mask
				if np.sum(diff) < keyframe_limit:
					diff = resize(np.mean(diff, axis=2), (self.height, self.width), anti_aliasing=True)
					over, under = diff > sensitivity_limit, diff <= sensitivity_limit
					diff[over], diff[under] = 255, 0.
					diff = resize(diff, (image.shape[0], image.shape[1]), anti_aliasing=False)
					# was the image already transparent?
					if transparent:
						image[:, :, 3] = diff
					else:
						image = np.dstack((image, diff))
					key_frames.append(False)
				else:
					key_frames.append(True)
			else:
				key_frames.append(True)
			# add transparency layer for keyframes also, for easier broadcasting
			if not self._is_transparent(image):
				image = np.dstack((image, np.ones((image.shape[0], image.shape[1]))))
			diff_images.append(image)

		# create a palette from all images if possible
		if self.regenerate_palette:
			warnings.warn("using regenerate_palette=True will result in flickering, as the palette will be regenerated for each image!", Warning)
		else:
			self._palette_from_list(diff_images)

		# merge keyframes and differences
		last = None
		for image, key in zip(diff_images, key_frames):
			current = self._convert(image, True, ~key)  # pyxelate keyframe / change
			if last is None:
				last = current
			else:
				# merge differences to previous images
				mask = ~np.logical_xor(last[:, :, 3], current[:, :, 3])
				last[mask] = current[mask]
			# generator
			yield last.copy()

	def _palette_from_list(self, images):
		"""Fit model to find palette using all images in list at once"""
		transparency = self._is_transparent(images[0])
		examples = []
		color_masks = []

		# sample from all images
		for image in images:
			examples.append(resize(image[:, :, :3], (16, 16), anti_aliasing=False).reshape(-1, 3).astype("int"))
			if transparency:
				color_masks.append(resize(images[0][:, :, 3], (16, 16), anti_aliasing=False))

		# concatenate to a single matrix
		examples = np.concatenate(examples)
		if transparency:
			# transparent colors should be ignored
			color_masks = np.concatenate(color_masks).ravel()
			examples = examples[color_masks >= self.alpha]
		self._fit_model(examples)

	def _fit_model(self, X):
		"""Fit model while suppressing warnings from sklearn"""
		converge = True
		with warnings.catch_warnings(record=True) as w:
			# fit model
			self.model.fit(X)
			if w and w[-1].category == ConvergenceWarning:
				warnings.filterwarnings('ignore', category=ConvergenceWarning)
				converge = False
		if not converge:
			warnings.warn("the model has failed to converge, try a different number of colors for better results!", Warning)
		self.is_fitted = True

	def _reduce(self, image):
		"""Apply convolutions on image ITER times and generate a smaller image
		based on the highest magnitude of gradients"""

		# self is visible to decorated function
		@adapt_rgb(each_channel)
		def _wrapper(dim):
			# apply median filter for noise reduction
			dim = median(dim, square(4))
			for n in range(self.ITER):
				h, w = dim.shape
				h, w = h // 2, w // 2
				flatten = view_as_blocks(dim, (2, 2)).reshape(-1, 2, 2)
				# bottleneck
				new_image = np.fromiter((self._reduce_conv(f) for f in flatten), flatten.dtype).reshape((h, w))
				if n < self.ITER - 1:
					dim = new_image.copy()
			return new_image

		return _wrapper(image)

	def _reduce_conv(self, f):
		"""The actual function that selects the right pixels based on the gradients  2x2 square"""
		return np.mean(f[self.SOLUTIONS[
			np.argmax(np.sum(np.multiply(self.CONVOLUTIONS, f.reshape(-1, 2, 2)).reshape(-1, 4), axis=1))]])

	def _dilate(self, image):
		"""Dilate semi-transparent edges to remove artifacts
		(unwanted edges, caused by transparent pixels having different colors)"""

		@adapt_rgb(each_channel)
		def _wrapper(dim):
			return dilation(dim, selem=square(4))

		# use dilated pixels for semi-transparent ones
		mask = image[:, :, 3]
		alter = _wrapper(image[:, :, :3])
		image[:, :, :3][mask < self.alpha] = alter[mask < self.alpha]
		return image

	@staticmethod
	def _fix_hist(image):
		"""Apply adaptive histogram"""
		image = equalize_adapthist(image) * 255 * 1.14  # empirical magic number
		image[image <= 8.] = 0.
		return image

	@staticmethod
	def _is_transparent(image):
		"""Returns True if there is an additional dimension for transparency"""
		return bool(image.shape[2] == 4)

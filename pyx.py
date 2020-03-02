import time
import argparse
import os
import numpy as np
from pyxelate import Pyxelate
import skimage
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt

def parse_arugments():
    parser = argparse.ArgumentParser(description='Pixelate an image or images in a directory.')
    parser.add_argument('-f', '--factor', required=False, metavar='factor', type=int, default=5, help='The factor by which the image should be downscaled. Defaults to 10.')
    parser.add_argument('-s', '--scaling', required=False, metavar='scaling', type=int, default=5, help='The factor by which the generated image should be upscaled. Defaults to 10.')
    parser.add_argument('-c', '--colors', required=False, metavar='colors', type=int, default=8, help='The amount of colors of the pixelated image. Defaults to 8.')
    parser.add_argument('-d', '--path', required=False, metavar='path', type=str, default='.', help='Path to the image or directory containing images for processing. Defaults to execution directory.')
    return parser.parse_args()

def get_filelist(path):
    if os.path.isdir(path):
        # generate file list
        (_, _, filenames) = next(os.walk(path))
        # print(filenames)
        return io.imread_collection(filenames)
    elif os.path.isfile(path):
        return io.imread_collection([path])
    else:
        print("path points to non image file")
        return None

def process(img, imgname, factor, colors, scaling):
    height, width, _ = img.shape
    p = Pyxelate(height // factor, width // factor, colors)
    pyxelated = p.convert(img)
    if scaling > 1:
        pyxelated = transform.resize(pyxelated, ((height // factor) * scaling, (width // factor) * scaling), anti_aliasing=False, mode='edge', preserve_range=True, order=0)
    io.imsave(imgname, pyxelated.astype(np.uint8))

def make_ouput_dir():
    path = os.getcwd() + "/" + str(int(time.time()))
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
        return ""
    else:
        return path

# Main programm
# =============

# parse arguments
args = parse_arugments()

# generate an image collection from the file or directory given
ic = get_filelist(args.path)

# try to create the output directory
output_dir = make_ouput_dir()

# if the output dir was created successfull, convert all images in the collection
if output_dir != "":
    n = 0
    for i in ic:
        n += 1
        filename = output_dir + "/" + str(n) + '.png'
        process(i, filename, args.factor, args.colors, args.scaling)
import argparse
import os
import skimage
import sys
import time
from pyxelate import Pyxelate
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
import numpy as np

def parse_arugments():
    parser = argparse.ArgumentParser(description='Pixelate an image or images in a directory.')
    parser.add_argument('-f', '--factor', required=False, metavar='factor', type=int, default=5, help='The factor by which the image should be downscaled. Defaults to 5.')
    parser.add_argument('-s', '--scaling', required=False, metavar='scaling', type=int, default=5, help='The factor by which the generated image should be upscaled. Defaults to 5.')
    parser.add_argument('-c', '--colors', required=False, metavar='colors', type=int, default=8, help='The amount of colors of the pixelated image. Defaults to 8.')
    parser.add_argument('-r', '--regenerate', required=False, metavar='regenerate', type=bool, default=True, help='Regenerate the palette for each image. Defaults to True.')
    parser.add_argument('-t', '--state', required=False, metavar='state', type=int, default=0, help='Sets the random state of the Bayesian Gaussian Mixture. Defaults to 0.')
    parser.add_argument('-p', '--path', required=False, metavar='path', type=str, default='.', help='Path to single image or directory containing images for processing. Defaults <cwd>.')
    parser.add_argument('-o', '--outpath', required=False, metavar='path', type=str, default='', help='Path to the directory where the pixelated images are stored. Defaults to <cwd>/pyxelated')
    return parser.parse_args()

def get_filelist(path):
    if os.path.isdir(path):
        # generate file list
        (_, _, filenames) = next(os.walk(path))
        return filenames
    elif os.path.isfile(path):
        return [path]
    else:
        print("path points to non image file")
        return None

def make_ouput_dir():
    path = os.getcwd() + "/pyxelated"

    try:
        os.mkdir(path)

    except FileExistsError:
        # if the path already exists just use it
        return path

    except OSError:
        print ("Creation of the directory %s failed" % path)
        sys.exit(1)

    else:
        return path

def load_image_from_path(path):
    try:
        image = io.imread(imagefile)
    except:
        return []
    else:
        return image

# Main programm
# =============

# parse arguments
args = parse_arugments()

# generate an image collection from the file or directory given
imagefiles = get_filelist(args.path)

# use the output directory defined by args
if args.outpath == "." and os.path.isdir(args.outpath):
    output_dir = args.outpath
# otherwise create one
else:
    output_dir = make_ouput_dir()

print("Writing files into %s" % output_dir)

# heigth and width are getting set per image, this are just placeholders
p = Pyxelate(1, 1, color=args.colors, regenerate_palette=args.regenerate, random_state=args.state)

# loop over all images in the directory
for imagefile in imagefiles:

    image = load_image_from_path(imagefile)

    # when the file is not an image just move to the next file
    if len(image) == 0:
        print("Skipping\t%s" % imagefile)
        continue

    print("Processing\t%s" % imagefile)
    outfile = output_dir + "/" + os.path.splitext(os.path.basename(imagefile))[0] + '.png'

    # get image dimensions
    height, width, _ = image.shape

    # apply the dimensions to pyxelate
    p.height = height // args.factor 
    p.width = width // args.factor
    pyxelated = p.convert(image)

    # scalue the image up if so requested
    if args.scaling > 1:
        pyxelated = transform.resize(pyxelated, ((height // args.factor) * args.scaling, (width // args.factor) * args.scaling), anti_aliasing=False, mode='edge', preserve_range=True, order=0)
    
    # finally save the image
    io.imsave(outfile, pyxelated.astype(np.uint8))
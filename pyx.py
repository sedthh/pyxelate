#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path
from pyxelate import Pyxelate
from numpy import uint8
from skimage import io
from skimage import transform


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Pixelate images in a directory.'
    )
    parser.add_argument(
        '-f', '--factor',
        required=False, metavar='factor', type=int, default=5,
        help='''The factor by which the image should be downscaled.
        Defaults to 5.'''
    )
    parser.add_argument(
        '-s', '--scaling',
        required=False, metavar='scaling', type=int, default=5,
        help='''The factor by which the generated image should be
        upscaled. Defaults to 5.'''
    )
    parser.add_argument(
        '-c', '--colors',
        required=False, metavar='colors', type=int, default=8,
        help='''The amount of colors of the pixelated image. Defaults
        to 8.'''
    )
    parser.add_argument(
        '-d', '--dither',
        required=False, metavar='dither', type=str_as_bool, nargs='?',
        default=True, help='Allow dithering. Defaults to True.'
    )
    parser.add_argument(
        '-a', '--alpha',
        required=False, metavar='alpha', type=float, default=.6,
        help='''Threshold for visibility for images with alpha channel.
        Defaults to .6.'''
    )
    parser.add_argument(
        '-r', '--regenerate_palette',
        required=False, metavar='regenerate_palette', type=bool,
        default=True, help='''Regenerate the palette for each image.
        Defaults to True.'''
    )
    parser.add_argument(
        '-t', '--random_state',
        required=False, metavar='random_state', type=int, default=0,
        help='''Sets the random state of the Bayesian Gaussian Mixture.

        Defaults to 0.'''
    )
    parser.add_argument(
        '-i', '--input',
        required=False, metavar='path', type=str, default='',
        help='''Path to single image or directory containing images for
        processing. Defaults <cwd>.'''
    )
    parser.add_argument(
        '-o', '--output',
        required=False, metavar='path', type=str, default='',
        help='''Path to the directory where the pixelated images are
        stored. Defaults to <cwd>/pyxelated'''
    )
    return parser.parse_args()


def str_as_bool(val):
    # interpret the string input as a boolean
    if val.lower() in ("false", "none", "no", "0"):
        return False
    return True


def get_file_list(path):
    path = Path(path)
    if path.is_dir():
        # get all files and directories except hidden
        tree = path.glob('**/[!.]*')
        # generate file list without directories
        file_names = [elm for elm in tree if elm.is_file()]
        return file_names
    elif path.is_file():
        return [path]
    else:
        print("Path points to non image file.")
        sys.exit(1)


if __name__ == "__main__":
    # get arguments and file list
    args = parse_arguments()
    image_files = get_file_list(args.input)

    # use the output directory defined by args
    if not args.output:
        output_dir = Path.cwd() / "pyxelated"
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing files to {output_dir}")

    # height and width are getting set per image, this are just placeholders
    p = Pyxelate( 1, 1, color=args.colors, dither=args.dither,
        alpha=args.alpha, regenerate_palette=args.regenerate_palette,
        random_state=args.random_state)

    # loop over all images in the directory
    for image_file in image_files:
        image = io.imread(image_file)
        base = str(image_file.stem) + ".png"

        # when the file is not an image just move to the next file
        if len(image) == 0:
            print(f"Skipping\t{image_file}")
            continue

        print(f"Processing\t{image_file}")

        outfile = output_dir / base

        # get image dimensions
        height, width, _ = image.shape

        # apply the dimensions to Pyxelate
        p.height = height // args.factor
        p.width = width // args.factor
        pyxelated = p.convert(image)

        # scale the image up if so requested
        if args.scaling > 1:
            pyxelated = transform.resize(pyxelated, (
                (height // args.factor) * args.scaling,
                (width // args.factor) * args.scaling),
                anti_aliasing=False, mode='edge',
                preserve_range=True, order=0
                )

        # finally save the image
        io.imsave(outfile, pyxelated.astype(uint8))

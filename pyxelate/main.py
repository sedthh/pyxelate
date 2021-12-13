#!/usr/bin/env python3

import argparse
import sys
from typing import List, Optional, Set, Tuple, Union

from skimage import io
from . import Pyx, Pal


def convert(args: argparse.Namespace):
    pyx = Pyx(
        height=args.height,
        width=args.width,
        factor=args.factor,
        upscale=args.upscale,
        depth=args.depth,
        palette=args.palette,
        dither=args.dither,
        sobel=args.sobel,
        alpha=args.alpha,
        boost=not args.noboost,
    )
    image = io.imread(args.INFILE)
    pyx.fit(image)
    new_image = pyx.transform(image)
    io.imsave(args.OUTFILE, new_image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INFILE", type=str, help="Input image filename.")
    parser.add_argument("OUTFILE", type=str, help="Output image filename.")
    parser.add_argument("--width", type=int, help="Output image width.", default=None)
    parser.add_argument("--height", type=int, help="Output image height.", default=None)
    parser.add_argument("--factor", type=int, help="Downsample factor.", default=1)
    parser.add_argument(
        "--upscale", type=int, help="Upscale factor for output pixels.", default=1
    )
    parser.add_argument(
        "--depth", type=int, help="Number of times to downscale.", default=1
    )
    parser.add_argument(
        "--palette",
        type=str,
        help="Number of colors in output palette, or a palette name. "
        f"Valid choices are: {list(Pal.__members__)}",
        default="8",
    )
    parser.add_argument(
        "--dither",
        type=str,
        help="Type of dithering to use.",
        default="none",
        choices=["none", "naive", "bayer", "floyd", "atkinson"],
    )
    parser.add_argument(
        "--sobel", type=int, help="Size of the Sobel operator.", default=3
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Alpha threshold for output pixel visibility.",
        default=0.6,
    )
    parser.add_argument(
        "--noboost",
        action="store_true",
        help="By default, adjust contrast and apply preprocessing on the image before "
        "transformation for better results. In case you see unwanted dark "
        "pixels in your image, use --noboost.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress logging output.")
    args = parser.parse_args()

    # The --palette arg can be an integer or a palette name.
    try:
        palette = int(args.palette)
    except ValueError:
        try:
            palette = Pal[args.palette]
        except KeyError:
            print(f"Bad value for --palette: {args.palette}")
            print(f"Valid choices are: {list(Pal.__members__)}")
            parser.print_usage()
            sys.exit(1)
    args.palette = palette

    if not args.quiet:
        print(f"Pyxelating {args.INFILE}...")

    convert(args)

    if not args.quiet:
        print(f"Wrote {args.OUTFILE}")


if __name__ == "__main__":
    main()

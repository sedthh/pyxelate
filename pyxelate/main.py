#!/usr/bin/env python3

import argparse
import sys
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from skimage import io
try:
    from . import Pyx, Pal, Vid
except ImportError:
    try:
        from pyxelate import Pyx, Pal, Vid
    except ImportError:
        from pal import Pal
        from pyx import Pyx
        from vid import Vid
    

def get_model(args: argparse.Namespace):
    return Pyx(
        height=args.height,
        width=args.width,
        factor=args.factor,
        upscale=args.upscale,
        depth=args.depth,
        palette=args.palette,
        dither=args.dither,
        sobel=args.sobel,
        alpha=args.alpha,
        svd=not args.nosvd,
    )

def convert(args: argparse.Namespace):
    pyx = get_model(args)
    image = io.imread(args.INFILE)
    pyx.fit(image)
    new_image = pyx.transform(image)
    io.imsave(args.OUTFILE, new_image)
    
def convert_sequence(args: argparse.Namespace):
    # get files from folder in order
    p = Path(args.INFILE)
    files = str(p.name)
    assert "%d" in files, "Input filename for sequences must contain %d to denote ordering!"
    candidates = [str(c.name) for c in list(p.parent.resolve().glob(f"*{p.suffix}"))]
    images, names, i = [], [], 0
    while True:
        check = [bool(re.search(r'\b' + files.replace("%d", f"[0]*?{i}") + r'\b', c)) for c in candidates]
        if np.any(check):
            name = p.parent.resolve() / candidates[np.argmax(check)]
            names.append(name)
            image = io.imread(name) 
            images.append(image)
        elif i > 1 or images:
            break
        i += 1
    all = len(images)
    assert all, f"No images found in {p.parent} that satisfied '{files}'"
    five_percent = max(1, all // 20)
    if not args.quiet:
        print(f"Found {all} '{p.suffix}' images in '{p.parent}'")    
    
    p = Path(args.OUTFILE)
    files = str(p.name)
    assert "%d" in files, "Output filename for sequences must contain %d to denote ordering!"
    # generate a new image sequence based on differences between them
    for i, (image, key) in enumerate(Vid(images, pad=args.pad, sobel=args.sobel, keyframe=args.keyframe, sensitivity=args.sensitivity)):
        if i == 0 or (key and args.refit):
            if not args.quiet:
                print(f"Fitting model on keyframe '{names[i]}'")    
            pyx = get_model(args)
            pyx.fit(image)
        # run the algorithm on the difference only
        image = pyx.transform(image)
        # save the pyxelated image
        file = str(p.parent / files.replace("%d", str(i)))
        io.imsave(file, image)
        if not args.quiet and i % five_percent == 0:
            print(f"Finished {i+1} out of {all} ({round((i + 1) / all * 100)}%)")    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INFILE", type=str, help="Input image filename. For sequence of images use: folder/img_%%d.png")
    parser.add_argument("OUTFILE", type=str, help="Output image filename. For sequence of images use: folder/output_%%d.png")
    parser.add_argument("--width", type=int, help="Output image width.", default=None)
    parser.add_argument("--height", type=int, help="Output image height.", default=None)
    parser.add_argument("--factor", type=int, help="Downsample factor.", default=None)
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
        "--nosvd",
        action="store_true",
        help="By default, apply truncated SVD on the image before"
        "transformation for better results. In case you want ignore "
        "this step, use --nosvd.",
    )
    
    # for animations
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Convert a series of images for animation "
        "by calculating the difference between them and only applying the algorithm "
        "on the differences between frames to reduce flicker."
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Fit the palette again for each new keyframe. "
        "By default only the very first image in the sequnce will be used for palette fitting. "
        "Only works for animations with --sequence."
    )
    parser.add_argument(
        "--pad", 
        type=int, 
        default=0,
        help="Cut black bars from the top and the bottom of the image sequence before conversion. "
        "Default value is 0 (0 lines will be removed from both top and bottom). "
        "Only works for animations with --sequence."
    )
    parser.add_argument(
        "--keyframe", 
        type=float, 
        default=.3,
        help="Percentage (0. - 1.) of average image difference needed to be considered a new keyframe."
        "Default value is 0.30. Only works for animations with --sequence."
    )
    parser.add_argument(
        "--sensitivity", 
        type=float, 
        default=.1,
        help="Percentage (0. - 1.) of RGB difference needed for a part of image to be considered different."
        "Default value is 0.10. Only works for animations with --sequence."
    )
    # other
    parser.add_argument("--quiet", action="store_true", help="Suppress logging output.")
    args = parser.parse_args()

    if args.sequence:
        if args.dither not in ("none", "naive"):
            raise ValueError(f"Only 'naive' dithering is available when converting a sequence of images! Please use '--dither naive' instead of '--dither {args.dither}'")
        if not args.nosvd and not args.quiet:
            print(f"TIP: consider using --nosvd with --sequence for increased performance")
         
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
    
    if args.sequence:
        convert_sequence(args)
    else:
        convert(args)
    
    if not args.quiet:
        print(f"Wrote {args.OUTFILE}")


if __name__ == "__main__":
    main()

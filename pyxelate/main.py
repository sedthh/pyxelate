#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from skimage import io
try:
    from . import Pyx, Pal, images_to_parts, parts_to_images
except ImportError:
    try:
        from pyxelate import Pyx, Pal, images_to_parts, parts_to_images
    except ImportError:
        from pal import Pal
        from pyx import Pyx
        from vid import images_to_parts, parts_to_images
    

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
        boost=not args.noboost,
    )

def convert(args: argparse.Namespace):
    pyx = get_model(args)
    image = io.imread(args.INFILE)
    pyx.fit(image)
    new_image = pyx.transform(image)
    io.imsave(args.OUTFILE, new_image)
    
def convert_sequence(args: argparse.Namespace):
    p = Path(args.INFILE)
    files = str(p.name)
    assert "%d" in files, "Input filename for sequences must contain %d to denote ordering!"
    candidates = [str(c.name) for c in list(p.parent.resolve().glob(f"*{p.suffix}"))]
    images, names, i = [], [], 0
    while True:
        check = files.replace("%d", str(i))
        if check in candidates:
            name = p.parent.resolve() / check
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
    
    # generate a new image sequence based on differences between them
    new_images, new_keys = [], []
    for i, (image, key) in enumerate(images_to_parts(images)):
        if i == 0 or (key and args.refit):
            if not args.quiet:
                print(f"Fitting model on keyframe '{names[i]}'")    
            pyx = get_model(args)
            pyx.fit(image)
        # run the algorithm on the difference only
        image = pyx.transform(image)
        # save the pyxelated image part for later
        new_images.append(image)
        new_keys.append(key)
        if not args.quiet and i % five_percent == 0:
            print(f"Finished {i+1} out of {all} ({round((i + 1) / all * 100)}%)")    
            
    if not args.quiet:
        print("Recreating pyxelated images...")
    # put the pyxelated parts back together
    p = Path(args.OUTFILE)
    files = str(p.name)
    assert "%d" in files, "Output filename for sequences must contain %d to denote ordering!"
    for i, image in enumerate(parts_to_images(new_images, new_keys)):
        file = str(p.parent / files.replace("%d", str(i)))
        io.imsave(file, image)
        if not args.quiet and i % five_percent == 0:
            print(f"Finished {i+1} out of {all} ({round((i + 1) / all * 100)}%)")    

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("INFILE", type=str, help="Input image filename. For sequence of images use: folder/img_%d.png")
    parser.add_argument("OUTFILE", type=str, help="Output image filename. For sequence of images use: folder/output_%d.png")
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
        "--nosvd",
        action="store_true",
        help="By default, apply truncated SVD on the image before"
        "transformation for better results. In case you want ignore "
        "this step, use --nosvd.",
    )
    parser.add_argument(
        "--noboost",
        action="store_true",
        help="By default, adjust contrast and apply preprocessing on the image before "
        "transformation for better results. In case you see unwanted dark "
        "pixels in your image, use --noboost.",
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
        "--square", 
        type=int, 
        default=2,
        help="Area of differences between images in a sequence are enlarged to this size."
        "Defalut value is 2. Only works for animations with --sequence."
    )
    parser.add_argument(
        "--keyframe", 
        type=float, 
        default=.66,
        help="Percentage (0. - 1.) of image difference needed to be considered a new keyframe."
        "Default value is 0.66. Only works for animations with --sequence."
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

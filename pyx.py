#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Command-line interface for Pyxelate. It simplifies batch processing
and provides user-friendly console output.

Usage:
pyx.py [-h help] [-i folder of input images or path to single image]
       [-o folder of output images] [-f scale down input image by factor]
       [-s scale up output image by factor] [-c colors] [-d dither] [-a alpha]
       [-r regenerate_palette] [-t random_state] [-w warnings] [-S sequence]
"""

import argparse
import sys
import warnings
import time as t
from pathlib import Path
from numpy import uint8
from skimage import io
from skimage import transform
from pyxelate import Pyxelate


def parse_arguments():
    """ Parses arguments and returns values. This function contains
    descriptions for the -h flag
    """
    parser = argparse.ArgumentParser(
        description='Pixelate images in a directory.'
    )
    parser.add_argument(
        '-f', '--factor',
        required=False, metavar='int', type=int, default=5, nargs='?',
        help='''The factor by which the image should be downscaled.
        Defaults to 5.'''
    )
    parser.add_argument(
        '-s', '--scaling',
        required=False, metavar='int', type=int, default=5, nargs='?',
        help='''The factor by which the generated image should be
        upscaled. Defaults to 5.'''
    )
    parser.add_argument(
        '-c', '--colors',
        required=False, metavar='2-32', type=int, default=8, nargs='?',
        help='''The amount of colors of the pixelated image. Defaults
        to 8.'''
    )
    parser.add_argument(
        '-d', '--dither',
        required=False, metavar='bool', type=str_as_bool, nargs='?',
        default=None, help='Allow dithering. Defaults to True.'
    )
    parser.add_argument(
        '-a', '--alpha',
        required=False, metavar='threshold', type=float, default=.6,
        nargs='?', help='''Threshold for visibility for images with
        alpha channel. Defaults to .6.'''
    )
    parser.add_argument(
        '-r', '--regenerate_palette',
        required=False, metavar='bool', type=str_as_bool, nargs='?',
        default=None, help='''Regenerate the palette for each image.
        Defaults to True.'''
    )
    parser.add_argument(
        '-t', '--random_state',
        required=False, metavar='int', type=int, default=0, nargs='?',
        help='''Sets the random state of the Bayesian Gaussian Mixture.
        Defaults to 0.'''
    )
    parser.add_argument(
        '-i', '--input',
        required=False, metavar='path', type=str, default='', nargs='?',
        help='''Path to single image or directory containing images for
        processing. Defaults <cwd>.'''
    )
    parser.add_argument(
        '-o', '--output',
        required=False, metavar='path', type=str, default='', nargs='?',
        help='''Path to the directory where the pixelated images are
        stored. Defaults to <cwd>/pyxelated'''
    )
    parser.add_argument(
        '-w', '--warnings',
        required=False, metavar='bool', type=str_as_bool, nargs='?',
        default=True, help='''Outputs non-critical library warnings.
        Defaults to True.'''
    )
    parser.add_argument(
        '-S', '--sequence',
        required=False, metavar='bool', type=str_as_bool, nargs='?',
        default=False, help='''Uses a separate function for converting
        image sequences. Defaults to False.'''
    )
    return parser.parse_args()


def str_as_bool(val):
    """ Interpret the string input as a boolean
    """
    if val.lower() in ("false", "none", "no", "0"):
        return False
    return True


# Exclude hidden files and directories
F_EXCLUDED = 0

def exclude_hidden(elm):
    """ A filter that returns a file if it's not hidden
    """
    global F_EXCLUDED
    if not any(e.startswith('.') for e in elm.parts):
        return elm
    F_EXCLUDED += 1
    return False


def with_extension(elm):
    """ A filter that returns only a file with extension
    """
    global F_EXCLUDED
    if elm.is_file() and '.' in elm.name:
        return elm
    F_EXCLUDED += 1
    return False


def get_file_list(path):
    """ Finds all the files recursively and filters them
    """
    path = Path(path)
    if path.is_file() and '.' in path.name:
        return [path]
    if path.is_dir():
        # Get all files and directories
        tree = list(path.glob('**/*'))
        # Filter files and directories
        tree = list(filter(exclude_hidden, tree))
        file_names = list(filter(with_extension, tree))
        return file_names
    print("Path points to " + s['red']("non image") + " file.")
    sys.exit(1)


def parse_path(file):
    """ Returns relative path, file name and extension
    """
    file, ext = str(file).rsplit('.', 1)
    inp = str(Path(args.input))
    if inp == '.':
        file = '/' + file
    try:
        path, file = file.rsplit('/', 1)
    except ValueError:
        path = ""
    if inp == str(file):
        path = ""
    path = path.replace(inp, "")
    path += '/' if path else ''
    return [path, file, ext]


# Define CLI colors
s = {
    'green': lambda txt: '\u001b[32m' + str(txt) + '\u001b[0m',
    'red': lambda txt: '\u001b[31m' + str(txt) + '\u001b[0m',
    'mag': lambda txt: '\u001b[35m' + str(txt) + '\u001b[0m',
    'dim': lambda txt: '\u001b[37;2m' + str(txt) + '\u001b[0m'
}

# Status bar logic
CUR_FILE = 0
WARN_CNT = 0
ERR_CNT = 0
TIME_IMG = []
T_SPLIT = []
AVG_LAST_VALS = 10
T_UP = '\x1b[1A'
T_ERASE = '\x1b[2K'
BAR_RMV = '\n' + T_ERASE + T_UP + T_ERASE


def sec_to_time(sec):
    """ Returns the formatted time H:MM:SS
    """
    mins, sec = divmod(sec, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:d}:{mins:02d}:{sec:02d}"


def bar_redraw(last=False):
    """ Updates the progress bar and current status
    """
    t_pass = round(t.process_time())
    i_cur = CUR_FILE
    # Print bar
    percent = round(i_cur / ALL_FILES * 100, 1)
    p_int = round(i_cur / ALL_FILES * 100) // 2
    pbar = "[ " + "•" * (p_int) + s['dim']("-") * (50 - p_int) + " ] "
    pbar += str(percent) + " %"
    print(pbar)
    # Print status
    stat = "Elapsed: " + sec_to_time(t_pass)
    stat += s['dim'](" | ") + "Remaining: "
    # Remaining time. Averaging requires at least 1 value
    if len(TIME_IMG) > 0 and not last:
        t_avg = sum(TIME_IMG) / len(TIME_IMG)
        rem = round(t_avg * (ALL_FILES - i_cur))
        stat += sec_to_time(rem)
    else:
        stat += "Calculating..." if not last else sec_to_time(0)
    stat += ("\nDone " + s['green'](str(i_cur)) + '/' +
             str(ALL_FILES) + s['dim'](" | "))
    if args.warnings:
        stat += "Warnings: " + s['mag'](str(WARN_CNT)) + s['dim'](" | ")
    stat += "Errors: " + s['red'](str(ERR_CNT))
    # Adding escape codes depending on the passed argument
    if last:
        stat = BAR_RMV + stat
    else:
        # Raise the carriage three lines up and return it
        stat = stat + T_UP * 3 + '\r'
    print(stat)


def print_warn(warn):
    """ Outputs text as a warning
    """
    if str(warn) and args.warnings:
        warn = str(warn).strip().capitalize()
        print(BAR_RMV + s['mag']("\tWarning: ") + warn)
        bar_redraw()


def print_err(err):
    """ Outputs text as an error
    """
    if str(err):
        err = str(err).strip().capitalize()
        print(BAR_RMV + s['red']("\tError: ") + err)
        bar_redraw()


def print_settings():
    """ Outputs the full list of settings
    """
    print("Pyxelate settings\n" +
          s['dim']("\tFactor: ") + "\t" +
          str(args.factor) + "\t" +
          s['dim']("\tScaling: ") + "\t" +
          str(args.scaling) + "\n" +
          s['dim']("\tColors: ") + "\t" +
          str(args.colors) + "\t" +
          s['dim']("\tDither: ") + "\t" +
          ("No", "Yes")[args.dither] + "\n" +
          s['dim']("\tAlpha channel: ") + "\t" +
          str(args.alpha) + "\t" +
          s['dim']("\tRegenerate: ") + "\t" +
          ("No", "Yes")[args.regenerate_palette] + "\n" +
          s['dim']("\tRandom state: ") + "\t" +
          str(args.random_state) + "\t" +
          s['dim']("\tSequence: ") + "\t" +
          ("No", "Yes")[args.sequence] + "\n")


if __name__ == "__main__":
    # Get arguments and file list
    args = parse_arguments()
    IMAGE_FILES = get_file_list(args.input)
    ALL_FILES = len(IMAGE_FILES)

    # Use the output directory defined by args
    if not args.output:
        OUTPUT_DIR = Path.cwd() / "pyxelated"
        OUTPUT_DIR.mkdir(exist_ok=True)
    else:
        OUTPUT_DIR = Path(args.output)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set the default settings depending on the type of conversion
    if args.dither is None:
        args.dither = (True, False)[args.sequence]
    if args.regenerate_palette is None:
        args.regenerate_palette = (True, False)[args.sequence]

    print_settings()

    # Get input and output paths
    INPUT_DIR = Path(args.input) if args.input else Path.cwd()

    # Display input path
    if "/" in str(INPUT_DIR):
        I_PATH, I_BASE = str(INPUT_DIR).rsplit('/', 1)
        print("Reading files from " + s['dim'](I_PATH + '/') + I_BASE)
    else:
        print("Reading files from " + s['dim'](str(Path.cwd())) +
              "/" + str(INPUT_DIR))

    # At least one relevant file is required to run
    print(end="\t")
    if IMAGE_FILES:
        print(s['green'](len(IMAGE_FILES)) + " relevant files found" +
              s['dim'](" | ") + s['red'](F_EXCLUDED) + " excluded" +
              "\n")
    else:
        print(s['red'](len(IMAGE_FILES)) + " relevant files found")
        sys.exit(1)

    # Display output path
    if "/" in str(OUTPUT_DIR):
        O_PATH, O_BASE = str(OUTPUT_DIR).rsplit('/', 1)
        print("Writing files to " + s['dim'](O_PATH + '/') + O_BASE)
    else:
        print("Writing files to " + str(OUTPUT_DIR))

    # Setup Pyxelate with placeholder dimensions 1x1
    p = Pyxelate(
        1, 1,
        color=args.colors,
        dither=args.dither,
        alpha=args.alpha,
        regenerate_palette=args.regenerate_palette,
        random_state=args.random_state
    )

    # Prepare images to convert sequence
    if args.sequence:
        SEQUENCE = [io.imread(file) for file in IMAGE_FILES]
        SEQ_IMAGE = p.convert_sequence(SEQUENCE)
        HEIGHT, WIDTH, _ = io.imread(IMAGE_FILES[0]).shape
        p.height = HEIGHT // args.factor
        p.width = WIDTH // args.factor

    for i, image_file in enumerate(IMAGE_FILES):
        # Make a time stamp to calculate remaining
        if i or not args.sequence:
            # Calculate the time difference between iterations
            T_SPLIT += [t.time()]
            if T_SPLIT[1:]:
                if len(TIME_IMG) == AVG_LAST_VALS:
                    del TIME_IMG[0]
                diff = round(T_SPLIT[1] - T_SPLIT.pop(0), 1)
                TIME_IMG.append(diff)
        else:
            # Skip the first iteration to generate sequence palette
            print("\tPreparing to convert image sequence...")

        # Get the path, file name, and extension
        base = str(image_file.stem) + ".png"
        outfile = OUTPUT_DIR / base
        f_path, f_name, f_ext = parse_path(image_file)

        # The file format must be supported by skimage
        try:
            image = io.imread(image_file)
        except ValueError:
            # When the file is not an image just move to the next file
            print(BAR_RMV + "\tSkipping " + s['red']("unsupported") +
                  ":\t" + s['dim'](f_path) + f_name + '.' +
                  s['red'](f_ext))
            bar_redraw()
            continue

        print(BAR_RMV + "\tProcessing image:\t" + s['dim'](f_path) +
              f_name + '.' + f_ext)

        # Redraw status bar
        bar_redraw()

        # Get current dimensions
        if args.sequence:
            # Get sequence dimensions
            height, width = HEIGHT, WIDTH
        else:
            # Get image dimensions
            height, width, _ = image.shape

        # Apply the dimensions to Pyxelate
        p.height = height // args.factor
        p.width = width // args.factor

        # Convert the image
        with warnings.catch_warnings(record=True) as w:
            try:
                if args.sequence:
                    # Convert image from sequence
                    pyxelated = next(SEQ_IMAGE)
                else:
                    # Convert a single image
                    pyxelated = p.convert(image)
            except KeyboardInterrupt:
                print(BAR_RMV + "\tCancelled with " + s['red']("Ctrl+C"))
                bar_redraw(last=True)
                sys.exit(0)
            except IndexError as e:
                # When the file is not an image just move to the next file
                ERR_CNT += 1
                print_err(e)
                bar_redraw()
                continue
            if w:
                WARN_CNT += 1
                print_warn(w.pop().message)

        # Scale the image up if so requested
        if args.scaling > 1:
            pyxelated = transform.resize(
                pyxelated, (
                    (height // args.factor) * args.scaling,
                    (width // args.factor) * args.scaling
                ),
                anti_aliasing=False, mode='edge',
                preserve_range=True, order=0
            )

        # Finally save the image
        with warnings.catch_warnings(record=True) as w:
            try:
                io.imsave(outfile, pyxelated.astype(uint8))
            except KeyboardInterrupt:
                print(BAR_RMV + "\tCancelled with " + s['red']("Ctrl+C"))
                bar_redraw(last=True)
                sys.exit(0)
            if w:
                WARN_CNT += 1
                print_warn(w.pop().message)

        CUR_FILE += 1

    bar_redraw(last=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
import warnings
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


# exclude hidden files and directories
f_all = 0
f_excluded = 0

def exclude_hidden(elm):
    global f_excluded
    if not any(i.startswith('.') for i in elm.parts):
        return elm
    f_excluded += 1
    return False


 # exclude directories and files without extension
def with_extension(elm):
    global f_excluded
    if elm.is_file() and '.' in elm.name:
        return elm
    f_excluded += 1
    return False


def get_file_list(path):
    global f_all
    path = Path(path)
    if path.is_dir():
        # get all files and directories
        tree = list(path.glob('**/*'))
        f_all = len(tree)
        # filter files and directories
        tree = list(filter(exclude_hidden, tree))
        file_names = list(filter(with_extension, tree))
        return file_names
    elif path.is_file() and '.' in path.name:
        return [path]
    else:
        print("Path points to " + red("non image") + " file.")
        sys.exit(1)


def parse_path(file):
    f_name, f_ext = str(file).rsplit('.', 1)
    f_name = f_name.replace(str(Path(args.input)) + '/', '')
    if '/' in f_name:
        f_path, f_name = f_name.rsplit('/', 1)
        f_path = '/' + f_path + '/'
    else:
        f_path = ""
    return [f_path, f_name, f_ext]


# define CLI colors and create functions
def style_def(func, ansi):
    exec(f'''def {func}(input):
        return "{ansi}" + str(input) + "\u001b[0m"
    ''', globals())

style_def('green', '\u001b[32m')
style_def('red', '\u001b[31m')
style_def('mag', '\u001b[35m')
style_def('dim', '\u001b[37;2m')


# status bar logic
cur_image = 0
time_img = []
t_up = '\x1b[1A'
t_erase = '\x1b[2K'
avg_last_vals = 10

def sec_to_time(sec):
    n, m, s = 0, 0, 0
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def bar_redraw(i_cur, i_all, t_pass, t_last):
    t_pass = round(t_pass)
    # print bar
    percent = round(i_cur / i_all * 100, 1)
    p_int = round(i_cur / i_all * 100) // 2
    b = "[ " + "•" * (p_int) + dim("-") * (50 - p_int) + " ] "
    b += str(percent) + " %"
    print(b)
    # print status
    r = "Done " + green(str(i_cur)) + '/' + str(i_all) + dim(" | ")
    r += "Elapsed: " + sec_to_time(t_pass) + dim(" | ") + "Remaining: "
    # averaging requires at least 1 value
    if len(t_last) > 0:
        t_avg = sum(t_last) / len(t_last)
        rem = round(t_avg * (i_all - i_cur))
        r += sec_to_time(rem)
    else:
        r += "Calculating..."
    print(r)
    # raise the carriage two lines up and return it
    print(t_up * 2 + '\r', end="")


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

    # get input and output paths
    input_dir = Path(args.input) if args.input else Path.cwd()
    o_path, o_base = str(output_dir).rsplit('/', 1)
    i_path, i_base = str(input_dir).rsplit('/', 1)

    # at least one relevant file is required to run
    if image_files:
        print(green(len(image_files)) + " relevant files found | " +
            red(f_excluded) + " excluded")
    else:
        print(red(len(image_files)) + " relevant files found")
        sys.exit(1)

    # display some information at the start
    print("Reading files from " + dim(i_path + '/') + i_base)
    print("Writing files to   " + dim(o_path + '/') + o_base)

    # height and width are getting set per image, this are just placeholders
    p = Pyxelate(1, 1, color=args.colors, dither=args.dither,
        alpha=args.alpha, regenerate_palette=args.regenerate_palette,
        random_state=args.random_state)

    # loop over all images in the directory
    for image_file in image_files:
        # get the path, file name, and extension
        base = str(image_file.stem) + ".png"
        outfile = output_dir / base
        f_path, f_name, f_ext = parse_path(image_file)
        cur_image += 1

        # get the time of the last iteration to calculate the remaining
        if 'img_end' in globals():
            if len(time_img) == avg_last_vals:
                del time_img[0]
                time_img.append(round(img_end - img_start, 1))
            else:
                time_img.append(round(img_end - img_start, 1))
        img_start = time.time()

        # the file format must be supported by skimage
        try:
            image = io.imread(image_file)
        except ValueError:
            # when the file is not an image just move to the next file
            print(t_erase + "\tSkipping " + red("unsupported") +
                ":\t" + dim(f_path) + f_name + '.' + red(f_ext))
            bar_redraw(cur_image, len(image_files), time.process_time(), time_img)
            continue

        print(t_erase + "\tProcessing image:\t" + dim(f_path) +
            f_name + '.' + f_ext)

        # redraw status bar
        bar_redraw(cur_image, len(image_files), time.process_time(), time_img)

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
        warnings.filterwarnings("error")
        try:
            io.imsave(outfile, pyxelated.astype(uint8))
        except UserWarning as e:
            re = "/".join([o_path, o_base, f_name]) + '.' + f_ext
            e = str(e).replace(re, "").strip()
            print(t_erase + mag("\tWarning: ") + "It " + e)

            warnings.filterwarnings("ignore")
            io.imsave(outfile, pyxelated.astype(uint8))

        img_end = time.time()

    print('\n')

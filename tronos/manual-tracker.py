#!/usr/bin/env python

# This is a VERY simple and naive manual tracker.
# If you need something more powerful, Fiji is
# absolutely recommended.

# Generic
import argparse
import csv
import numpy as np
import sys
import os

# Imaging and visualization
from PIL import Image
import matplotlib.pyplot as plt


# Functions
def open_image(froute):
    try:
        img = Image.open(froute)
        img.load()
        print(img.n_frames)
        print(img)
        return img
    except:
        print("Unable to load image")


def on_press(event):
    trajs.append([event.x, event.y])


def build_trajs(img):
    for i in range(img.n_frames):
        try:
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
            plt.connect("button_press_event", on_press)
            plt.show()
            img.seek(i)
        except EOFError:
            # Not enough frames in img
            break


def csv_save(trajs, csvname):
    with open(csvname, "a+") as f:
        for i, pos in enumerate(trajs):
            f.writelines("{},{},{}".format(i, pos[0], pos[1]))


# Argument parsing
def cmdline_args():

    p = argparse.ArgumentParser(
        description="""
        Tronos VERY simple manual tracker
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--source",
        nargs="+",
        help="Source TIFF file to manually process. CSV results will be saved to selected directory",
    )
    p.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot routes when manual tracking completed for each trajectory",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = cmdline_args()
    trajs = []
    for fname in args.source:
        trajs = []
        img = open_image(fname)
        build_trajs(img)
        csvname = os.path.basename(os.path.splitext(fname)[0]) + "_trajs.csv"
        csv_save(trajs, csvname)
    pass

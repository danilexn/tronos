#!/usr/bin/env python3

# This program reads a dv file and transforms (credits to ...)
# Also implements in-house training and prediction of asci location,
# particle detection and tracking (tracker & neural)
#
# Using Detecto (torch) and trackpy. Particle detection supplied thanks to Fiji

import struct
from PIL import Image, ImageDraw, ImageFilter
import sys
import numpy as np
import cv2
import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm
import argparse
import os
import threading

# Owned
import tracker
import neural

__author__ = "Daniel Leon-Perinan"
__copyright__ = "Copyright 2020"
__credits__ = ["Daniel Leon"]
__license__ = "MIT License"
__version__ = "0.1.0"
__maintainer__ = "Daniel Leon"
__email__ = "daniel@ilerod.com"
__status__ = "Dev"
header = """  
  ______ ____   ____   _   __ ____  _____
 /_  __// __ \ / __ \ / | / // __ \/ ___/
  / /  / /_/ // / / //  |/ // / / /\__ \ 
 / /  / _, _// /_/ // /|  // /_/ /___/ / 
/_/  /_/ |_| \____//_/ |_/ \____//____/  
                                         """


class progParameters:
    dvExtendedHeaderSize = 0
    dvImageWidth = 0
    dvImageHeight = 0
    dvNumOfImages = 0
    dvPixelType = 0
    dvTimePoints = 0
    dvImageDataOffset = 0
    dvNumberZSections = 0
    dvExtendedHeaderNumInts = 0
    dvExtendedHeaderNumFloats = 0
    offset = 0
    size = 0


def read_header(fname):
    p = progParameters()
    f = open(fname, "rb")
    dvdata = f.read(256)
    dvExtendedHeaderSize = struct.unpack_from("<I", dvdata, 92)[0]
    # endian-ness test
    if not struct.unpack_from("<H", dvdata, 96)[0] == 0xC0A0:
        print("unsupported endian-ness")
        exit(1)

    progParameters.dvImageWidth = struct.unpack_from("<I", dvdata, 0)[0]
    progParameters.dvImageHeight = struct.unpack_from("<I", dvdata, 4)[0]
    progParameters.dvNumOfImages = struct.unpack_from("<I", dvdata, 8)[0]
    progParameters.dvPixelType = struct.unpack_from("<I", dvdata, 12)[0]
    progParameters.dvTimePoints = struct.unpack_from("<H", dvdata, 180)[0]
    progParameters.dvImageDataOffset = 1024 + dvExtendedHeaderSize
    progParameters.dvNumberZSections = (
        progParameters.dvNumOfImages / progParameters.dvTimePoints
    )
    print(
        "W: {}, H: {}, N: {}, T: {}, Offset: {}, Z: {}, Pts: {}".format(
            progParameters.dvImageWidth,
            progParameters.dvImageHeight,
            progParameters.dvNumOfImages,
            progParameters.dvPixelType,
            progParameters.dvImageDataOffset,
            progParameters.dvNumberZSections,
            progParameters.dvTimePoints,
        )
    )

    if progParameters.dvPixelType != 6:
        print("unsupported pixel type")
        exit(1)

    progParameters.dvExtendedHeaderNumInts = struct.unpack_from("<H", dvdata, 128)[0]
    progParameters.dvExtendedHeaderNumFloats = struct.unpack_from("<H", dvdata, 130)[0]
    sectionSize = 4 * (
        progParameters.dvExtendedHeaderNumFloats
        + progParameters.dvExtendedHeaderNumInts
    )
    sections = dvExtendedHeaderSize / sectionSize

    print(
        "Sections: {}, SectionSize: {}, Extended: {}".format(
            sections, sectionSize, progParameters.dvExtendedHeaderSize
        )
    )

    if sections < progParameters.dvNumOfImages:
        print("number of sections is less than the number of images")
        exit(1)
    sections = progParameters.dvNumOfImages

    progParameters.offset = progParameters.dvImageDataOffset
    progParameters.size = progParameters.dvImageWidth * progParameters.dvImageHeight * 2
    return f, p


def save_im(f, offset, factor, p):
    f.seek(offset)
    dvdata = f.read(p.size)
    im = Image.frombuffer(
        "I;16", [p.dvImageWidth, p.dvImageHeight], dvdata, "raw", "I;16", 0, 1
    )
    im = Image.eval(im, lambda x: x * factor)
    im = im.convert("L")
    return im

# From https://stackoverflow.com/questions/48213278/implementing-otsu-binarization-from-scratch-python/50796152
def otsu(gray, p):
    pixel_number = p.dvImageWidth * p.dvImageHeight
    mean_weigth = 1.0 / pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[
        1:-1
    ]:  # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        Wb = np.sum(his[:t]) * mean_weigth
        Wf = np.sum(his[t:]) * mean_weigth
        mub = np.mean(his[:t])
        muf = np.mean(his[t:])
        value = Wb * Wf * (mub - muf) ** 10
        if value > final_value:
            final_thresh = t
            final_value = value
    gray[gray > final_thresh] = 255
    gray[gray <= final_thresh] = 0
    return gray


def stack_project(f, p):
    imlist = []
    mtlist = []
    stack = np.zeros((20, p.dvImageWidth, p.dvImageHeight))
    for i in tqdm(range(int(p.dvTimePoints))):
        for j in range(int(p.dvNumberZSections)):
            im = save_im(f, p.offset, 1, p)
            stack[j, :, :] = im
            p.offset += p.size
        im_max = np.uint8(np.max(stack, axis=0))
        mtlist.append(im_max)
        imlist.append(Image.fromarray(im_max, "L"))
    imlist[0].save(
        args.workdir[0] + "/%s.tif" % (p.outname),
        compression="tiff_deflate",
        save_all=True,
        append_images=imlist[1:],
    )
    if args.otsu:
        print("Generating OTSU projection")
        imlist_otsu = []
        for image in tqdm(mtlist):
            imlist_otsu.append(Image.fromarray(otsu(image, p), "L"))
        imlist_otsu[0].save(
            args.workdir[0] + "/%s_PRJ.tif" % (p.outname),
            compression="tiff_deflate",
            save_all=True,
            append_images=imlist_otsu[1:],
        )
    f.close()


def tiff_convert(f, p):
    imlist = []
    for i in tqdm(range(p.dvNumOfImages)):
        green = save_im(f, p.offset, 0.5870, p)  # 12 bit data in range 0-4095
        red = save_im(f, p.offset, 0.2989, p)  # double the intensity of the red channel
        blue = save_im(f, p.offset, 0.1140, p)
        im = Image.merge("RGB", (red, green, blue))
        imlist.append(im)
        p.offset += p.size
        # draw = ImageDraw.Draw(im)
        # draw.text((10, 10),elapsed_times[i*2])
    if args.individual:
        for i in range(p.dvNumOfImages):
            imlist[i].save(args.workdir[0] + "/%s_%i.jpeg" % (p.outname, i))
    else:
        imlist[0].save(
            args.workdir[0] + "/%s.tif" % (p.outname),
            save_all=True,
            append_images=imlist[1:],
        )
    f.close()


def thread_function(fname):
    # print("[INFO] Opening ".format(os.path.basename(fname)))
    f, p = read_header(fname)
    p.outname = os.path.basename(os.path.splitext(fname)[0])
    if args.number:
        p.dvNumOfImages = int(args.number[0])
    # print("[INFO] Header from {}.dv acquired. Continue to processing.".format(p.outname))
    if args.stack:
        # print("[INFO] Z-Stacking will occur now")
        stack_project(f, p)
        # print("[INFO] Done Z-Stacking")
    elif args.convert:
        tiff_convert(f, p)
    print("[INFO] Completed processing of {}.dv to .tiff".format(p.outname))
    if args.track:
        ctrl = ""
        if args.otsu:
            ctrl = "_PRJ"
        tracker.detect_and_track(args.workdir[0] + "/%s%s" % (p.outname, ctrl))


# Make parser object
def cmdline_args():
    p = argparse.ArgumentParser(
        description="""
        This is a test of the command line argument parser in Python.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--source", nargs="+", help="Source DV file to regenerate")
    p.add_argument("--workdir", nargs="+", help="Destination directory")
    p.add_argument(
        "--train",
        nargs="+",
        help="Train neural model with specified route (labelImg format required)",
    )
    p.add_argument(
        "--detect", nargs="+", help="Detect using specified neural network model"
    )
    p.add_argument(
        "--model", nargs="+", help="Specifies folder which serves as model IO"
    )
    p.add_argument("--stack", action="store_true", help="Enable Z Stack processing")
    p.add_argument(
        "--convert", action="store_true", help="Enable dv to tiff conversion"
    )
    p.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Output verbosity",
    )
    p.add_argument("-t", "--track", action="store_true", help="Track route")
    p.add_argument("-o", "--otsu", action="store_true", help="Track route")
    p.add_argument("-i", "--individual", action="store_true", help="Track route")
    p.add_argument(
        "-n",
        "--number",
        nargs=1,
        help="Number of images to process per file. Be cautious, if file size exceeded, a thread exception may occur.",
    )

    return p.parse_args()


if __name__ == "__main__":
    print(header)
    print("Welcome to TRONOS {} by {}".format(__version__, __author__))
    print("[INFO] Parsing arguments...")
    args = cmdline_args()
    if args.source:
        for fname in args.source:
            th = threading.Thread(target=thread_function, args=(fname,))
            th.start()
    if args.train:
        model = neural.m_training(args.train[0], args.train[1], ["ascus", "normal"])
        neural.model_save(args.model[0], model)
    if args.detect:
        neural.m_predict(args.detect, args.model[0], args.workdir[0])

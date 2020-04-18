#!/usr/bin/env python3
# Future
from __future__ import division, unicode_literals, print_function

# Generic
import sys, traceback, subprocess, os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

# Script-specific
from skimage import io
import pims
import trackpy as tp
from io import StringIO


def generate_macro(path_to_imagej_macro):
    """
    Fiji macro generation and saving
    """
    ImageJ_macro = """

        open(getArgument());
        run("Set Measurements...", "area centroid center perimeter fit shape feret's area_fraction stack redirect=None decimal=3");
        run("Analyze Particles...", "size=500-10000 show=Outlines display clear stack");
        close()

        """

    with open(path_to_imagej_macro, "w") as macro:
        macro.write(ImageJ_macro)


def resolve_particles(file_name, path_to_imagej_macro):
    """
    Invokes Fiji for image processing
    Returns a DataFrame with positional and morphological features
    """

    # WARNING: this is for macOS. If running on Linux, this works for me:
    # ./home/[YOUR_USERNAME]/Fiji.app/ImageJ-linux64
    # That is, change the first line in command to suit your OS.
    # Windows has not been tested at all.

    command = (
        "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
        + " --headless --console -macro '"
        + path_to_imagej_macro
        + "' "
        + file_name
    )
    p = subprocess.Popen(
        command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    output = StringIO(p.stdout.read().strip().decode("UTF-8"))

    df = pd.read_csv(output, sep="\t")
    return df


def adapt_dataframe(df):
    """
    Transforms Fiji output in a way that is understood by trackpy.
    """
    df_dest = DataFrame()
    # It does not mind to change column names in morphological descriptors
    # as it does not affect particle linking.
    # TODO: function to change again to proper names.
    df_dest["y"] = df["YM"]
    df_dest["x"] = df["XM"]
    df_dest["mass"] = df["Minor"]
    df_dest["size"] = df["Area"]
    df_dest["ecc"] = df["Solidity"]
    df_dest["signal"] = df["Perim."]
    df_dest["raw_mass"] = df["Major"]
    df_dest["ep"] = df["Circ."]
    df_dest["frame"] = df["Slice"]
    return df_dest


def detect_and_track(file_name):
    # Data parsing, to be passed to Fiji
    print("[TRONOS]->[FIJI] Passing data for file {}".format(file_name))
    path_to_imagej_macro = "/tmp/tronos_count.imj"
    generate_macro(path_to_imagej_macro)
    # Fiji analysis
    print("[FIJI] Analyzing particles with default parameters")
    f_i = resolve_particles(file_name + ".tif", path_to_imagej_macro)
    print("[FIJI]->[TRONOS] Result parsing and transformation")

    # DataFrame adaptation for particle linking-tracking
    f = adapt_dataframe(f_i)
    print("[TRONOS] Trajectory linking")
    tp.quiet(suppress=True)
    t = tp.link(
        f, 100, memory=2
    )  # TODO: function implements arguments for easy manipulation of those parameters
    print("[TRONOS] Filtering trajectories")
    t1 = tp.filter_stubs(t, 170)

    # Compare the number of particles in the unfiltered and filtered data.

    print("[TRONOS] Track filter statistics")
    print("\tBefore:", t["particle"].nunique())
    print("\tAfter:", t1["particle"].nunique())
    t1.to_csv(file_name + "_trajs.csv")
    print("[TRONOS] Trajectories were successfully saved to file trajectories.csv")

#!/usr/bin/env python3
# Future
from __future__ import division, unicode_literals, print_function

# Generic
import sys, traceback, subprocess, os
import pandas as pd
from pandas import DataFrame, Series

# Script-specific
import trackpy as tp
from io import StringIO


def generate_macro(path, minsize, maxsize):
    """
    Fiji macro generation and saving
    """
    source = """
        open(getArgument());
        run("Set Measurements...", "area centroid center perimeter fit shape feret's area_fraction stack redirect=None decimal=3");
        run("Analyze Particles...", "size={}-{} show=Outlines display clear stack");
        close()
        """.format(minsize, maxsize)

    with open(path, "w") as macro:
        macro.write(source)


def resolve_particles(file_name, path):
    """
    Invokes Fiji for image processing
    Returns a DataFrame with positional and morphological features
    """

    # WARNING: this is for macOS and Linux
    # Windows has not been tested at all.

    command = ""

    if sys.platform == "darwin":
        if (not os.path.isdir("/Applications/Fiji.app")):
            print("[TRONOS] [ERROR] Could not find FIJI installed in your macOS system")

        command = (
            "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
            + " --headless --console -macro '"
            + path
            + "' "
            + file_name
            )
    else:
        if (not os.path.isdir("~/Fiji.app")):
            print("[TRONOS] [ERROR] Could not find FIJI installed in your Linux system")

        command = (
            "~/Fiji.app/ImageJ-linux64"
            + " --headless --console -macro '"
            + path
            + "' "
            + file_name
        )
    try:
        p = subprocess.Popen(
            command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
    except Exception:
        print("[TRONOS] [ERROR] Could not execute particle analysis")

    output = StringIO(p.stdout.read().strip().decode("UTF-8"))

    p.terminate()

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


def detect_and_track(filename, max_dist, min_frames, null_dist, sizes = [500, 10000]):
    # Data parsing, to be passed to Fiji
    print("[TRONOS]->[FIJI] Passing data for file {}".format(filename))
    path_macro = "/tmp/tronos_count.imj"
    generate_macro(path_macro, sizes[0], sizes[1])
    # Fiji analysis
    print("[FIJI] Analyzing particles with default parameters")
    f_i = resolve_particles(filename + ".tif", path_macro)
    print("[FIJI]->[TRONOS] Result parsing and transformation")

    # DataFrame adaptation for particle linking-tracking
    f = adapt_dataframe(f_i)
    print("[TRONOS] Trajectory linking")
    tp.quiet(suppress=True)
    t = tp.link(
        f, max_dist, memory=null_dist
    )  # TODO: function implements arguments for easy manipulation of those parameters
    print("[TRONOS] Filtering trajectories")
    t1 = tp.filter_stubs(t, min_frames)

    # Compare the number of particles in the unfiltered and filtered data.

    print("[TRONOS] Track filter statistics")
    print("\tBefore:", t["particle"].nunique())
    print("\tAfter:", t1["particle"].nunique())
    # Uncomment to change column names again to source format
    t1.columns = ["y", "x", "minor", "area", "convexity", "perimeter", "major", "circularity", "frame", "particle"]
    t1["particle"] = t1.particle.astype(str)
    t1["particle"] = t1["particle"] + "_{}".format(filename)
    t1.to_csv(filename + "_trajs.csv")
    print("[TRONOS] Trajectories were successfully saved to file {}".format(filename + "_trajs.csv"))
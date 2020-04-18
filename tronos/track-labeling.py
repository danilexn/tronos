#!/usr/bin/env python

# Reads a set of CSV files, with its respective brightfield coordinates, and transforms into a linked CSV file

from tqdm import tqdm
import pandas as pd
import argparse
import os
import numpy as np


def link_particles_to_coordinates(part, coord):
    part["label"] = "_none_"
    for i in range(len(part.index)):
        x, y = part["x"][i], part["y"][i]
        for _, r in coord.iterrows():
            if r["x_1"] <= x and r["x_2"] >= x and r["y_1"] <= y and r["y_2"] >= y:
                if part["label"][i] == "_none_":
                    part.loc[i, "label"] = r["label"] + ";"
                else:
                    part.loc[i, "label"] += r["label"] + ";"
    return part


def add_column_to_table():
    pass


def calculate_size_from_points(df):
    colnames = [c for c in df.columns if c != "label"]
    return colnames


def filename_transform(path, old, new, coordir):
    dname = os.path.dirname(os.path.realpath(path))
    if len(coordir) != 0:
        dname = coordir
    fname = os.path.basename(os.path.realpath(path))

    for i, o in enumerate(old):
        fname = fname.replace(o, new[i])
    return dname + "/" + fname


def remove_extension(fname):
    new_fname = os.path.splitext(fname)[0]
    return new_fname


def open_file(filename):
    try:
        df = pd.read_csv(filename, header=0)
        return True, df
    except:
        return False, []


# Argument parsing function
def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description="""
        This program allows labeling of CSV files out from Tronos-ReadFile, 
        labeled with respect to a coordinates file from a Tronos-ReadFile:Tracking
        pipeline. Model labels will be added as a column, as well as features derived from
        the object position and size (XY, center, size), if applicable.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("-s", "--source", nargs="+", help="Source *.[C|T]SV files to link")
    p.add_argument(
        "-o",
        "--outdir",
        nargs="+",
        help="Destination directory for the new *_linked.[C|T]SV files",
    )
    p.add_argument(
        "-i",
        "--coordir",
        nargs="+",
        help="Source coordinates directory for the *_coordinates.[C|T]SV files",
    )
    p.add_argument("-d", "--namedel", nargs="+", help="Filename deletion key")
    p.add_argument("-a", "--nameadd", nargs="+", help="Filename addition key")
    p.add_argument(
        "-c", "--category", nargs="+", help="Category of the analysis, as a label"
    )
    p.add_argument(
        "-f", "--features", nargs="+", help="Coordinate file features to be added"
    )
    p.add_argument(
        "-l",
        "--linkvar",
        nargs="+",
        help="Array of variables to be linked between coordinates and particles",
    )
    p.add_argument(
        "-t", "--sort", action="store_true", help="Enable sorting of the final table"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = cmdline_args()
    for filename in tqdm(args.source):
        correct, df_partic = open_file(filename)
        if not correct:
            break
        df_partic["category"] = args.category[0]
        df_partic.drop_duplicates(subset=None, keep="first", inplace=False)
        correct, df_coords = open_file(
            filename_transform(filename, args.namedel, args.nameadd, args.coordir[0])
        )
        if not correct:
            break
        new_df_partic = link_particles_to_coordinates(df_partic, df_coords)
        if args.sort:
            new_df_partic = new_df_partic.sort_values(by=["particle"], kind="mergesort")
        new_df_partic.to_csv(
            remove_extension(filename_transform(filename, [], [], args.outdir[0]))
            + "_linked.csv",
            header=True,
            index=False,
        )
    pass

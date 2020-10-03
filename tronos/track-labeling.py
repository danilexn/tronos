#!/usr/bin/env python

# Reads a set of CSV files, with its respective brightfield or meiotic stage
# coordinates and annotation, transforming them into linked CSV files

from tqdm import tqdm
from rtree import index
from rtree.index import Rtree
import pandas as pd
import argparse
import os
import numpy as np


NUM_TO_LABEL = {}


def link_particles_to_coordinates(part, coord, lp):
    part[args.feature] = "_none_"
    # Query particles against the index
    for index, row in part.iterrows():
        part.loc[index, args.feature] = ";".join(list(map(NUM_TO_LABEL.get, coord.intersection(
            (float(row[lp[0]]), float(row[lp[2]]), float(row[lp[1]]), float(row[lp[3]]))
        ))))
    return part


def create_index(coord, lc):
    p = index.Property()
    idx = index.Index(properties=p)
    # Create the index for feature coordinates
    for i, row in coord.iterrows():
        NUM_TO_LABEL[i] = row["label"]
        idx.add(i, [float(row[lc[0]]), float(row[lc[2]]), float(row[lc[1]]), float(row[lc[3]])])

    return idx


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
        type=str,
        help="Source coordinates directory for the *_coordinates.[C|T]SV files",
    )
    p.add_argument("-d", "--namedel", nargs="+", help="Filename deletion key")
    p.add_argument("-a", "--nameadd", nargs="+", help="Filename addition key")
    p.add_argument(
        "-c", "--category", help="Name for the analysis (example: wt_cell)"
    )
    p.add_argument(
        "--feature",
        type=str,
        help="Feature name, added as the column name",
        default="label",
    )
    p.add_argument(
        "--clink",
        nargs=4,
        help="Coordinate names for the R-tree; must be the area bounds",
        default = ["x_1", "x_2", "y_1", "y_2"]
    )
    p.add_argument(
        "--plink",
        nargs=4,
        help="Particle coordinate names in same order as --clink; may be a centroid",
        default = ["x", "x", "y", "y"]
    )
    p.add_argument("--first", action="store_true", help="Save destination coordinates from single-first frame")
    p.add_argument(
        "--sortby", type=str, help="Enable sorting of the final table", default=None
    )
    return p.parse_args()


if __name__ == "__main__":
    args = cmdline_args()
    for filename in tqdm(args.source):
        correct, df_partic = open_file(filename)
        if not correct:
            print("[ERROR] Could not open {}".format(filename))
            continue
        df_partic["category"] = args.category
        df_partic.drop_duplicates(subset=None, keep="first", inplace=False)
        correct, df_coords = open_file(
            filename_transform(filename, args.namedel, args.nameadd, args.coordir)
        )
        if not correct:
            print("[ERROR] Could not find coordinate file for {}".format(filename))
            continue

        df_partic_time = df_partic.sort_values(by=["frame"], kind="mergesort")
        result = pd.DataFrame()

        for t in df_partic_time["frame"].unique():
            NUM_TO_LABEL = {}
            df_coords_time = df_coords[df_coords.loc[:]["frame"] == (t - 1)]
            coords = create_index(df_coords_time, args.clink)
            if args.first:
                df_partic_coords = link_particles_to_coordinates(df_partic_time, coords, args.plink)
                result = result.append(df_partic_coords)
                break
            df_partic_coords = link_particles_to_coordinates(df_partic_time[df_partic_time.loc[:]["frame"] == t], coords, args.plink)
            result = result.append(df_partic_coords)

        if args.sortby is not None:
            try:
                result = result.sort_values(by=[args.sort], kind="mergesort")
            except Exception:
                print(
                    "ERROR: Could not sort the dataframe by the specified column {}".format(
                        args.sortby
                    )
                )

        result.to_csv(
            remove_extension(filename_transform(filename, [], [], args.outdir[0]))
            + "_linked.csv",
            header=True,
            index=False,
        )
    pass
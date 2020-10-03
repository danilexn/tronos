#!/usr/bin/env python3

# This is the main analysis workflow point for Tronos
# R is required, as packages can be considered a good standard
# Libs segclust2d, forecast, tsmp, pracma and Mclust are required
# Make sure that R is properly installed in your machine, and
# packages are available at their latest stable version


# Generic/Built-in
import numpy as np
import pandas as pd
import scipy
import os, sys
import argparse
import json
import requests
import datetime
from scipy import signal
from scipy import stats

# Other Libs
from statsmodels.tsa.stattools import grangercausalitytests
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2 import rinterface
from rpy2.robjects.vectors import StrVector
import uuid

# Global variables {code}
coords = ["x", "y"]
morphs = ["ecc", "raw_mass"]
time_varnames = ["time", "frame", "frame.1"]
non_varnames = ["particle", "category", "label"] + time_varnames
len_rnd = 181
num_rnd = 100

# R functions
segment_fct = """
        function(y, maxs){
            library(segclust2d)
            sequence <- data.frame(Y=as.matrix(y))
            segments <- suppressMessages(segmentation(sequence, lmin = 10, Kmax = maxs, seg.var = c("Y"), subsample_by = 1, scale.variable = TRUE))
            name <- paste0(segments$Kopt.lavielle," segments")
            statuses <- c((segments[["outputs"]][[name]][["segments"]][["end"]]), (segments[["outputs"]][[name]][["segments"]][["begin"]]))
            return(statuses)
        }
        """

arima_fct = """
        function(y){
            library(forecast)
            fittingarima <- auto.arima(y)
            return(fittingarima[["arma"]])
        }
    """

spec_fct = """
        function(tse, sampling, nd){
            library(pracma)
            y <- diff(tse, ndiffs = nd)
            del <- 1/sampling
            y.spec <- spectrum(y,log="yes",span=2,plot=FALSE)
            spx <- y.spec$freq/del
            spy <- 2*y.spec$spec
            pks <- findpeaks(spy, npeaks=3, threshold=0.1, sortstr=TRUE)
            mat <- matrix(0, nrow = 2, ncol = 3)
            try({mat <- matrix(nrow = 2, ncol = dim(pks)[1])
            mat[1,] <- 1/spx[pks[,2]]
            mat[2,] <- 1/spx[pks[,3]] - 1/spx[pks[,4]]})
            return(mat)
        }
    """

sigmoid_fct = """
        function(y){
            library(drc)
            y <- (y - min(y))/(max(y)-min(y))
            x <- 1:length(y)
            model <- drm(y ~ x, fct = L.4())
            as.vector(model[["parmMat"]])
        }
    """

lowess_fct = """
        function(y, fn){
            return(lowess(ts(y), f = fn)$y)
        }
    """

random_fct = """
        function(l){
            xx <- arima.sim(model = list(order = c(0, 1, 0)), n = l)
            return(as.numeric(xx))
        }
    """

motif_fct = """
        function(ts, wl){
            library(tsmp)
            library(rjson)
            {
                sink("/dev/null");
                mp <- invisible(suppressMessages(tsmp(ts, window_size = wl)))
                motifs <- invisible(suppressMessages(find_motif(mp, n_motifs = 10, n_neighbors = 20)))
                sink();
                return(toJSON(motifs$motif))
            }
        }
    """

class_fct = """
        function(ts){
            library(rjson)
            {
                sink("/dev/null");
                library(mclust)
                ts_classes <- invisible(suppressMessages(Mclust(ts, window_size = wl)))
                sink();
            }
            return(toJSON(ts_classes$parameters))
        }
    """

# R function initialization
segmentation = robjects.r(segment_fct)
autoarima = robjects.r(arima_fct)
spectrum = robjects.r(spec_fct)
sigmoid_fit = robjects.r(sigmoid_fct)
lowess = robjects.r(lowess_fct)
random = robjects.r(random_fct)
motif_discovery = robjects.r(motif_fct)
class_discovery = robjects.r(class_fct)


# Class declaration
# DataSet object
class DataSet:
    def __init__(self, df, name, timevar):
        self.name = name
        self.variables = []
        self.timevar = timevar

    def from_file(self, df):
        variableNames = [c for c in df.columns if c not in non_varnames]
        currentvNames = [c.name for c in self.variables]
        # df = df.sort_values(by=['particle'], kind = "mergesort")
        # TODO: instead of this method, implement sum of classes
        for v in variableNames:
            var = variable(v, [])
            if v in currentvNames:
                var = [var for var in self.variables if var.name == v][0]
            categoryNames = np.unique(df["category"].tolist())
            currentcNames = [c.name for c in var.categories]
            for c in categoryNames:
                cat = category(c, [])
                if c in currentcNames:
                    cat = [cat for cat in var.categories if cat.name == c][0]
                particleNames = np.unique(df["particle"].tolist())
                for p in particleNames:
                    cat.particles.append(
                        particle(
                            p,
                            df[v][df["particle"] == p][df["category"] == c].tolist(),
                            df[self.timevar][df["particle"] == p][
                                df["category"] == c
                            ].tolist(),
                            df["label"][df["particle"] == p][
                                df["category"] == c
                            ].tolist()[0],
                        )
                    )
                if c not in currentcNames:
                    var.categories.append(cat)
            if v not in currentvNames:
                self.variables.append(var)

    def add_variable(self, v):
        self.variables.append(v)


# Particle object
class particle:
    def __init__(self, name, sequence, time, label):
        self.name = name
        self.sequence = sequence
        self.time = time
        self.length = len(self.sequence)
        self.analysis = []
        self.label = label
        self.segments = []

        if self.name == "random":
            self.random_trajectories()
        elif self.name == "white":
            self.white_trajectories()

    def extract_segments(self):
        positions = [0]
        n_segments = analysis(["S"], [0])

        if args.depth[0] == "segmented":
            try:
                positions = [
                    c
                    for c in segmentation(
                        lowess(
                            robjects.FloatVector(self.seqReduction(int(args.w[0]))),
                            float(args.F[0]),
                        ),
                        int(args.maxseg[0]),
                    )
                ]
                n_segments.output = [int(len(positions) / 2)]
                for s in range(n_segments.output[0]):
                    segment = particle(
                        "_seg_" + str(s) + "_:" + str(self.name),
                        self.sequence[
                            positions[s + n_segments.output[0]] * 4 : positions[s] * 4
                        ],
                        [],
                        self.label,
                    )
                    segment.add_analysis(analysis(["S"], [0]))
                    self.add_segment(segment)
            except:
                log("Could not segment particle {}".format(self.name))

        self.add_analysis(n_segments)

    def add_segment(self, p):
        self.segments.append(p)

    def analyze_particle(self):
        shapiro = analysis(["SW"], [0])
        arimaorders = analysis(["p", "q", "d", "sp", "sq", "sd"], [0, 0, 0, 0, 0, 0])
        maxpeak_period = analysis(
            ["P1", "bw1", "P2", "bw2", "P3", "bw3"], [0, 0, 0, 0, 0, 0]
        )
        model_params = analysis(["a", "b", "c", "d"], [0, 0, 0, 0])
        try:
            arimaorders.output = [
                c for c in autoarima(robjects.FloatVector(self.sequence))
            ][0:6]
        except:
            log("Could not modelize with ARIMA for current particle")

        try:
            shapiro.output = [stats.shapiro(p.seqReduction(int(args.w[0])))[1]]
        except:
            log("Could not assay normality for current particle")

        try:
            model_params.output = [
                v
                for v in sigmoid_fit(
                    robjects.FloatVector(normalize(self.seqReduction(int(args.w[0]))))
                )
            ]
        except:
            log("Could not get model parameters for current particle")

        try:
            # Add output according to obtained, transformed into flat list
            maxpeak_period.output = (
                np.array(
                    spectrum(
                        robjects.IntVector(self.sequence),
                        float(args.sampling[0]),
                        int(args.diffs[0]),
                    )
                )
                .flatten("F")
                .tolist()
            )
            maxpeak_period.name = [
                "P{}".format(int(1 + i / 2))
                if i % 2 == 0
                else "bw{}".format(int((i + 1) / 2))
                for i in range(len(maxpeak_period.output))
            ]
            # Add maximum peaks to the list (append to the beginning)
            maxpeak_period.output = [max(maxpeak_period.output)] + maxpeak_period.output
            maxpeak_period.name = ["Pmax"] + maxpeak_period.name
        except:
            log("Could not detect any peak for current particle")

        self.add_analysis(shapiro, arimaorders, model_params, maxpeak_period)

    def normalize(self):
        data = self.sequence
        normalized = (data - min(data)) / (max(data) - min(data))
        return normalized

    def add_analysis(self, *args):
        for a in args:
            self.analysis.append(a)

    def print_analysis(self, c):
        SEPARATOR = "\t"
        header = "Name\tID\tLabel\tLen\t"
        line = "{}\t{}\t{}\t{}\t".format(c.name, self.name, self.label, self.length)
        for a in self.analysis:
            header += SEPARATOR.join(a.name) + "\t"
            line += (
                SEPARATOR.join(
                    [
                        str(c)
                        if not isinstance(c, float)
                        else str("{:+.03f}".format(c))
                        for c in a.output
                    ]
                )
                + "\t"
            )

        return header, line

    def random_trajectories(self):
        self.sequence = [c for c in random(180)]
        self.length = len(self.sequence)

    def quality(self):
        stdev_diff = np.std(self.diff())
        return stdev_diff

    def diff(self):
        return np.diff(self.sequence)

    def white_trajectories(self):
        mean = 0
        std = 1
        x = np.random.normal(mean, std, size=len_rnd)
        self.sequence = x

    def seqReduction(self, interval):
        amplitude = []
        if args.reduction[0] == "windowed":
            for i in range(interval, self.length, interval):
                temp = self.diff()[(i - interval + 1) : i]
                amplitude.append(sum(abs(temp)))
        elif args.reduction[0] == "spectral":
            _, amplitude = signal.periodogram(p.sequence, float(args.sampling[0]))
        else:
            amplitude = p.sequence
        return amplitude


class analysis:
    def __init__(self, name, output):
        self.name = name
        self.output = output


class category:
    def __init__(self, name, particles):
        self.name = name
        self.particles = particles
        self.amplitudes = []
        self.segments = []
        self.frequencies = []
        self.motifs = []

    def whole_sequence(self, scaled=False):
        wholeseq = []
        for p in self.particles:
            wholeseq += p.sequence if not scaled else normalize(p.sequence)
        return wholeseq

    def analyze_motifs(self, motif_file, class_file):
        seq = robjects.FloatVector(self.whole_sequence(scaled=True))
        # Change scaled = True to False if you do not want this discovery
        # to be homogeneous across series.
        # TODO: provide as a selectable argument (NOT URGENT)
        write_json(motif_file, motif_discovery(seq, int(args.W[0])))
        write_json(class_file, class_discovery(seq))


class variable:
    def __init__(self, name, categories):
        self.name = name
        self.categories = categories


# Utilities functions
def open_file(filename):
    df = pd.read_csv(filename, header=0)
    return df


def check_header(df):
    columns = [c for c in df.columns if c in time_varnames]
    if len(columns) > 0:
        return True, columns[0]
    return False, ""


def write_csv(route, *args):
    header = args[0][0]
    line = args[0][1]
    to_print = ""
    if not os.path.exists(route):
        to_print += header + "\n"
    to_print += line + "\n"
    with open(route, "a+") as csvfile:
        csvfile.writelines(to_print)


def write_json(route, json_string):
    with open(route, "w") as f:
        f.writelines(json_string)


def prettify_json(*args):
    for f in args:
        parsed = ""
        with open(f) as f_r:
            parsed = json.load(f_r)
        with open(f, "w") as f_w:
            f_w.writelines(json.dumps(parsed, indent=4, sort_keys=True))


def log(message, console=True):
    constructed = "[{}] - {}".format(datetime.datetime.now(), message)
    if console:
        print(constructed)
    else:
        return constructed


# Statistics and maths functions
def sigmoid(x, a, b, c, d):
    return b + (c - b) / (1 + np.exp(a * (x - d)))


def normalize(data):
    normalized = [(float(i) - max(data)) / (max(data) - min(data)) for i in data]
    return normalized


# Argument parsing function
def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(
        description="""
        Tronos particle statistical analysis program.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--source", nargs="+", help="Source CSV file to process")
    p.add_argument(
        "-o", "--outdir", nargs="+", help="Output directory to write CSV result"
    )
    p.add_argument(
        "-r", "--random", nargs=1, help="Number of random trajectories to generate"
    )
    p.add_argument(
        "-p",
        "--significance",
        nargs=1,
        default=["0.95"],
        help="Significance threshold for the statistical analysis",
    )
    p.add_argument("-m", "--maxseg", nargs=1, default=["4"], help="Maximum segments")
    p.add_argument(
        "-d",
        "--diffs",
        nargs=1,
        default=["1"],
        help="Number of differences to apply to time series",
    )
    p.add_argument(
        "-s", "--sampling", nargs=1, default=["0.01666"], help="Sampling rate in Hz"
    )
    p.add_argument(
        "--depth",
        nargs=1,
        default=["segmented"],
        help="Indicate whether analysis will be performed as a [whole], or [segmented]",
    )
    p.add_argument(
        "--reduction",
        nargs=1,
        default=["windowed"],
        help="Indicate the reduction method as [windowed], [spectral] or [none]",
    )
    p.add_argument(
        "-w", nargs=1, default=["4"], help="Window length size for windowed amplitude"
    )
    p.add_argument(
        "-W", nargs=1, default=["12"], help="Window length size for motif discovery"
    )
    p.add_argument(
        "-F",
        nargs=1,
        default=["0.16"],
        help="F for LOWESS smoothing (greater F, greater smoothing)",
    )
    p.add_argument(
        "--motif",
        action="store_true",
        help="Enable automatic motif analysis, as a whole",
    )

    return p.parse_args()


# Main routine
if __name__ == "__main__":
    # Parse console arguments
    args = cmdline_args()
    # Random elements parsing
    if args.random:
        num_rnd = int(args.random[0])
    # Create the DataSet, null.
    ds = DataSet([], str(uuid.uuid4()), [])
    # Iterate over the list of files supplied
    log("Loading {} files into memory with ID {}".format(len(args.source), ds.name))
    for fname in args.source:
        log("Loaded file {}".format(fname))
        # Create a DataFrame with the file contents
        df = open_file(fname)
        # Check whether the header in the file follows a proper format (else, next file)
        time_column_checked, time_column_name = check_header(df)
        if not time_column_checked:
            log("Incorrect file header at {}".format(fname))
            break
        # Fill a DataSet, with the variables, categories, particles and segments in the current file.
        ds.timevar = time_column_name
        ds.from_file(df)
    # Create a random variable. Sequence number parsed as argument.
    ds.add_variable(
        variable(
            name="random",
            categories=[
                category(
                    "random",
                    particles=[
                        particle(str(e), [c for c in random(len_rnd)], [], "_none_")
                        for e in range(1, num_rnd)
                    ],
                )
            ],
        )
    )

    # Iterate over variables
    log("Analyzing variables {}".format(", ".join([v.name for v in ds.variables])))
    for v in ds.variables:
        # Iterate over variable categories
        for c in v.categories:
            log(
                "Analyzing category {} with {} particle (s)".format(
                    c.name, len(c.particles)
                )
            )
            # Generate output file name
            out_file_name = args.outdir[0] + "/" + c.name + "_" + v.name + ".tsv"
            # Iterate over initially generated particles
            for p in c.particles:
                # Generating segments
                p.extract_segments()
                p.analyze_particle()
                # Printing analysis
                write_csv(out_file_name, p.print_analysis(c))
                # Analyze particle segments
                # log("Particle {} yielded {} segment(s)".format(p.name, len(p.segments))) # (v level 2)
                for s in p.segments:
                    s.analyze_particle()
                    write_csv(out_file_name, s.print_analysis(c))
            # Motif and class analysis once per-particle analysis finished
            if args.motif:
                log("Analyzing motifs and classes in {},{}".format(v.name, c.name))
                # Generate file names
                motif_file = (
                    args.outdir[0] + "/" + c.name + "_" + v.name + "_motifs.json"
                )
                class_file = (
                    args.outdir[0] + "/" + c.name + "_" + v.name + "_classes.json"
                )
                # Begin per-class analysis
                c.analyze_motifs(motif_file, class_file)
                # This could be optional
                prettify_json(motif_file, class_file)
    pass

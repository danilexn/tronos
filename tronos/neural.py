#!/usr/bin/env python3

# Reads a folder of images and processes it as a Neural Network model, using the Detecto library
# for both training and predicting. Credits to the RTX 2070S for the model supplied in the repo.

from detecto import core, utils, visualize
from detecto.core import Model
from tqdm import tqdm
import tifffile
import mrc
import argparse
import os
import pandas as pd
from PIL import Image


WRONG_DIM = "Dimensions are wrong. You must provide a 2D image"


def m_training(modelTags, trainRoute, testRoute, e=15, r=0.01):
    """
    Training of model with specified images and XML annotations (labelImg)
    """
    dataset = core.Dataset(trainRoute)
    val_dataset = core.Dataset(testRoute)
    model = core.Model(modelTags)

    model.fit(dataset, val_dataset, epochs=e, learning_rate=r, verbose=True)
    return model


def m_predict(modelTags, modelRoute, image):
    """
    Object detection through trained model
    """
    result = pd.DataFrame()
    model = Model.load(modelRoute, modelTags)
    frames = 1
    if len(image.shape) == 4:
        frames = image.shape[0]

    for f in range(frames):
        predictions = model.predict(image)
        labels, boxes, scores = predictions
        boxes_np = boxes.numpy()
        scores_np = scores.numpy()
        result = result.append(
            pd.DataFrame(
                {
                    "x_1": boxes_np[:,0],
                    "x_2": boxes_np[:,2],
                    "y_1": boxes_np[:,1],
                    "y_2": boxes_np[:,3],
                    "label": labels,
                    "scores": scores_np
                }
            )
        )
        result["frame"] = f

    return result[result["scores"] >= args.score]


def open_tif(route):
    # Read the image using tifffile
    image = tifffile.imread(route)

    # Check that the image is a 4D stack
    if len(image.shape) >= 4:
        raise ValueError(WRONG_DIM)

    return image


def open_dv(route):
    # Load the .dv file into memory with mrc
    image = mrc.imread(route)

    # Parse and understand the header
    header = image.Mrc.hdr
    nt, nw = header.NumTimes, header.NumWaves
    _, _, nsecs = header.Num
    nz = int(nsecs / nt / nw)

    if nz > 1:
        raise ValueError(WRONG_DIM)

    return image


def cmdline_args():
    p = argparse.ArgumentParser(
        description="""
        Neural training and prediction for image labeling. 2D images with time dimension 
        may be provided as queries or training-validation set (.tiff and .dv are valid)
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--query",
        nargs="+",
        help="""Query tiff or dv file(s), only 2D images
    with time, not 4D stacks can be used for detection nor training!""",
    )
    p.add_argument("--workdir", nargs="+", help="Destination directory")
    p.add_argument(
        "--train",
        nargs="+",
        help="""Train neural model with specified route (labelImg format required)\n
        Please, specify both training set and validation set folders
        """,
    )
    p.add_argument("--rate", type=int, help="Learning rate for training", default=0.01)
    p.add_argument("--epochs", type=int, help="Epochs for training", default=15)
    p.add_argument("--score", type=float, help="Score threshold to consider boxes", default=0.0)
    p.add_argument(
        "--categories",
        nargs="+",
        help="Categories as specified in the labelImg annotation",
    )
    p.add_argument(
        "--model",
        type=str,
        help="Where and how to save the model (specify .pth extension)",
    )

    return p.parse_args()


opener = {".tif": open_tif, ".tiff": open_tif, ".dv": open_dv}

if __name__ == "__main__":
    args = cmdline_args()
    if args.train:
        model = m_training(
            args.categories, args.train[0], args.train[1], args.rate, args.epochs
        )
        model.save(args.model)
    else:
        print("Predicting {} in specified image".format(args.model))
        for q in tqdm(args.query):
            extension = os.path.splitext(q)[1]
            q_im = opener[extension](q)
            prediction = m_predict(args.categories, args.model, q_im)
            prediction.to_csv("{}_coords.csv".format(os.path.splitext(q)[0]), header=True, index=False)

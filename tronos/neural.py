#!/usr/bin/env python3

# Reads a folder of images and processes it as a Neural Network model, using the Detecto library
# for both training and predicting. Credits to the RTX 2070S for the model supplied in the repo.

from detecto import core, utils, visualize
from detecto.core import Model
from detecto.visualize import show_labeled_image
from tqdm import tqdm
import torch
import os


def m_training(trainRoute, testRoute, modelTags):
    """
    This function allows training of model with specified images and XML files
    """
    dataset = core.Dataset(trainRoute)
    val_dataset = core.Dataset(testRoute)
    model = core.Model(modelTags)

    model.fit(dataset, val_dataset, epochs=15, learning_rate=0.01, verbose=True)
    return model


def m_predict(predictRoute, modelRoute, workdir):
    """
    This function allows object detection through prediction based on the trained model
    """
    print("Predicting asci in {} images".format(len(predictRoute)))
    for singleFile in tqdm(predictRoute):
        model = Model.load(modelRoute, ["ascus", "normal"])
        image = utils.read_image(singleFile)
        predictions = model.predict(image)
        labels, boxes, scores = predictions
        boxes_np = boxes.numpy()
        with open(
            workdir
            + "/"
            + os.path.basename(os.path.splitext(singleFile)[0])
            + "_coordinates.csv",
            "a",
        ) as fcsv:
            fcsv.writelines("x_1,y_1,x_2,y_2,label\n")
            for i, l in enumerate(labels):
                x_1, y_1, x_2, y_2 = boxes_np[i]
                fcsv.write("{},{},{},{},{}\n".format(x_1, y_1, x_2, y_2, l))
    return labels, boxes, scores


def model_save(modelRoute, model):
    """
    This function allows saving the trained model for later usage
    """
    model.save(modelRoute)

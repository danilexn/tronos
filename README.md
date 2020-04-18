# TRONOS
*TRacking of Nuclear OScillations*

This repository contains code and example input-output files for the automated pipeline used to assay nuclear oscillation patterns during meiotic prophase in <em>S. pombe</em>. This allows 2D or 3D reconstruction, as soon as there is enough S/N ratio.


## Scripts
Contained in folder *tronos*:
(Instructions for each script is given with the <em>--help</em> flag)
1. *tronos.py*: contains the code for image processing, blob detection and tracking. Some parameters must be changed in the own code. Only maximum projection is used before binarization, although sum could be implemented.
2. *tracker.py* & *neural.py*: contain the code referenced in *tronos.py* for object-blob track reconstruction and the neural-based workflow (training and prediction) used for asci detection in brightfield images.
3. *track-labeling.py*: must be called to link and label tracks according to the type of cell (["normal", "ascus"]). The model has a very high True Positive rate, with FN being a bit worse. Some manual verification is recommended.
4. *manual-tracker.py*: manual tracking of blobs.
5. *status-utilities.py*: some unused but maybe useful functions for math calculations.
6. *particle-analysis.py*: calculation of all statistics useful for pattern detection and analysis, per particle and per segment (number of segments, length, ARIMA coefficients, spectral richness and sigmoid fitting).

## Sample files
Contained in folder *samples/\**:
1. *linked/*: contains .csv files with reconstructed trajectories and its labeling
2. *particle_results/*: contains .tsv files with statistical analysis of each particle as input
3. *unlinked/*: contains .csv files with the labeling or annotation for each brightfield file, as the coordinates.

## Tests
Tests folder is empty by now; unit tests will be used upon the project gets more complex and remaining classes are implemented.

## Models
Contains neural models (as the weighting file) in pth format, used during brightfield labeling.
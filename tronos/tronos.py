#!/usr/bin/env python3


import struct
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import sys, traceback, subprocess, os
import threading
import tracker
import tifffile
import mrc

__author__ = "Daniel Leon-Perinan"
__copyright__ = "Copyright 2020"
__credits__ = ["Daniel Leon"]
__license__ = "MIT License"
__version__ = "0.2.0"
__maintainer__ = "Daniel Leon"
__email__ = "dleoper@upo.es"
__status__ = "Dev"


TEMP_THRESH = "/tmp/tronos_threshold.imj"
header = """
  ______ ____   ____   _   __ ____  _____
 /_  __// __ \ / __ \ / | / // __ \/ ___/
  / /  / /_/ // / / //  |/ // / / /\__ \ 
 / /  / _, _// /_/ // /|  // /_/ /___/ /  
/_/  /_/ |_| \____//_/ |_/ \____//____/   
                                         """

# Legacy compatibility with 0.1.0 Tronos
# Reading of a dv files; credits to Lambert et al. (also for mrc)
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


def reference_convert(f, p):
    imlist = []
    if args.first:
        imNum = 1
    else:
        imNum = p.dvNumOfImages
    for i in tqdm(range(imNum)):
        green = save_im(f, p.offset, 0.5870, p)  # 12 bit data in range 0-4095
        red = save_im(f, p.offset, 0.2989, p)  # double the intensity of the red channel
        blue = save_im(f, p.offset, 0.1140, p)
        im = Image.merge("RGB", (red, green, blue))
        imlist.append(im)
        p.offset += p.size
        # draw = ImageDraw.Draw(im)
        # draw.text((10, 10),elapsed_times[i*2])
    if args.individual:
        for i in range(imNum):
            imlist[i].save(args.workdir[0] + "/%s_%i.tiff" % (p.outname, i))
    else:
        imlist[0].save(
            args.workdir[0] + "/%s.tif" % (p.outname),
            save_all=True,
            append_images=imlist[1:],
        )
    f.close()


class TronosImage:
    def load_image(self, route):
        extension = os.path.splitext(route)[1]
        self.wavelengths, size, shape, image = opener[extension](route)
        try:
            self.nt, self.nz, self.nw, self.nx, self.ny = shape
        except (Exception):
            os.system(Exception)
            exit(1)
        self.image = image
        self.pxx, self.pxy, self.pxz = size

    def __init__(self, route):
        self.load_image(route)
        super().__init__()

    def frame(self, t, w):
        if self.nt <= 1 and self.nw > 1:
            im = self.image[w, 0 : self.nz, :, :]
        elif self.nw <= 1:
            if len(self.image.shape) == 5:
                im = self.image[t, 0 : self.nz, w, :, :]
            elif self.nt > 1 and self.nz > 1:
                im = self.image[t, 0 : self.nz, :, :]
            elif self.nt > 1:
                im = self.image[t, :, :]
            else:
                im = self.image[0 : self.nz, :, :]
        else:
            if self.image.shape[1] == self.nz:
                im = self.image[t, 0 : self.nz, w, :, :]
            elif self.image.shape[1] == self.nw:
                im = self.image[t, w, 0 : self.nz, :, :]
        return im


def open_tif(route):
    """[summary]
    Args:
        route (string): route for the .tiff file being processed
    Raises:
        ValueError: when wavelengths (even if one) are not provided
        ValueError: when a 4D stack is not provided
    Returns:
        tuple: complying the order in the Image class, returns the tuple
        of dimensions (TZWXY), as well as pixel size and other features
    """
    # Read the image using tifffile
    image = tifffile.imread(route)

    if len(image.shape) < 3:
        raise ValueError("[ERROR] You have to provide, at least, a 3D stack. Exiting...")

    # Check that the image is a 4D stack
    if len(image.shape) == 3:
        nt, nx, ny = image.shape
        nw, nz = 1, 1
        waves = []
    elif len(image.shape) == 4:
        nt, nz, nx, ny = image.shape
        nw = 1
        waves = []
    else:
        if args.waves == None:
            raise ValueError("No wavelengths were provided. Please refer to --help")
        waves = args.waves
        nt, nz, nw, nx, ny = image.shape

    return waves, [1, 1, 1], [nt, nz, nw, nx, ny], image


def open_dv(route):
    """Opens a DeltaVision .dv image stack, and generates a tuple of 
    settings using mrc
    Args:
        route (string): route for the .dv file being processed
    Returns:
        tuple: complying the order in the Image class, returns the tuple
        of dimensions (TZWXY), as well as pixel size and other features
    """
    # Load the .dv file into memory with mrc
    image = mrc.imread(route)

    # Parse the header
    header = image.Mrc.hdr

    # Parse the number of timepoints and wavelengths
    nt, nw = header.NumTimes, header.NumWaves

    # Parse the number of XY pixels (image resolution)
    nx, ny, nsecs = header.Num

    # Parse the number of z planes
    nz = int(nsecs / nt / nw)

    # Parse the pixel and stack sizes in microns
    pxx, pxy, pxz = header.d[0:3]
    return header.wave[0:4], [pxx, pxy, pxz], [nt, nz, nw, nx, ny], image


def join_spinning(route, channels):
    """Joins a set of Spinning-Disk folder
    structure to a .tif file consisting of a Z-Stack.
    You must provide the route as a foldername, it is
    mandatory that inside the folder, channel names (subfolders)
    must be provided. This set of subfolders contains the Z-stacks
    in individual files for each timepoint. They will be joined in a
    two step process: 
        1. A stack for a single channel is reconstructed in memory
        2. The stack is merged into a single file
    This process takes place in RAM: please, make sure enough memory is 
    available!
    Args:
        route ([string]): folder name containing files to reconstruct a movie
        channels ([list]): list of channel names (same as subsequent folder names)
    Returns:
        [string]: The generated 4D stack file route
    """
    # Creates an empty list that will contain the final stack
    movieStack = []

    # Iterates over the user-provided list of channels
    for channel in channels:  # argument provided as spinning

        # Creates an empty list that will contain the channel stack
        channelStack = []

        # Route for the channel stack (individual timepoitns)
        channelPath = os.path.join(route, channel)

        # Individual files are parsed. Timepoint order in filename (..._t1, ..._tn)
        files = [
            os.path.join(channelPath, f)
            for f in os.listdir(channelPath)
            if os.path.isfile(os.path.join(channelPath, f))
        ]

        # Select timepoints based on *_{tN}.TIF notation
        frames = [
            int("".join(os.path.basename(os.path.splitext(f)[0]).split("_")[-1][1:]))
            for f in files
        ]
        # Sort the timepoints
        frames.sort()

        # Select a basename to iterate over timepoints
        filenameBase = "_".join(os.path.splitext(files[0])[0].split("_")[:-1])

        # Store the exact file extension
        extension = os.path.splitext(files[1])[1]

        # Iterating over files, construct the channelStack list
        for frame in frames:
            image = tifffile.imread("{}_t{}{}".format(filenameBase, frame, extension))
            channelStack.append(image)

        # Stores the channel inside the final movie
        movieStack.append(channelStack)

    # Transposes the array to maintain the classical TCZXY ordering (4D stack)
    # TODO: this is a bit wanky, will be fixed
    movieStack = np.array(movieStack)
    movieStack = np.swapaxes(movieStack, 0, 1)
    movieStack = np.swapaxes(movieStack, 1, 2)

    # Stores the file with a new, derived name
    filename = "{}_stack.tif".format(route)
    tifffile.imsave("{}".format(filename), movieStack)

    # Returns the generated stack image fileroute
    return filename


def max_projection(im, minimum=0, maximum=-1):
    """Returns the Z maximum projection of an image, between the
    provided minimum and maximum slices
    Args:
        im (np.array): image data as numpy array, any format. 4D stack!
        minimum (int, optional): minimum slice in Z projection. Defaults to 0.
        maximum (int, optional): maximum slice in Z projection. Defaults to -1.
    Returns:
        np.array: z maximum projected image
    """
    stack = []
    for t in tqdm(range(0, im.nt)):
        frame = np.array(im.frame(t, 0))
        stack.append(np.max(frame[minimum:maximum, :, :], axis=0))

    imResult = np.array(stack)
    return imResult


def macro_threshold(output, method, blur, crop):
    """
    Fiji macro generation for thresholding
    """

    sourceBlur = ""
    sourceCrop = ""
    if blur:
        sourceBlur = """
            run("Gaussian Blur...", "sigma=2 stack");
            setAutoThreshold("{} dark");
            //run("Threshold...");
            setOption("BlackBackground", true);
            run("Convert to Mask", "method={} background=Dark calculate black");
            """.format(method, method)

    if crop:
        sourceCrop = """
        makeRectangle({}, {}, {}, {});
        run("Crop");
        """.format(crop[0],crop[1],crop[2],crop[3])

    source = """
        open(getArgument());
        run("16-bit");
        {}
        setAutoThreshold("{} dark");
        //run("Threshold...");
        setOption("BlackBackground", true);
        run("Convert to Mask", "method={} background=Dark calculate black");
        {}
        run("8-bit");
        saveAs("Tiff", "{}");
        close()
        """.format(
        sourceCrop, method, method, sourceBlur, output
    )

    with open(TEMP_THRESH, "w") as macro:
        macro.write(source)


def threshold(inputimage, outputimage, method="MaxEntropy", blur=False, crop = []):
    """
    Invokes Fiji for image processing
    Saves image with max entropy thresholding
    """

    macro_threshold(outputimage, method, blur, crop)

    if sys.platform == "darwin":
        if (not os.path.isdir("/Applications/Fiji.app")):
            print("[TRONOS] [ERROR] Could not find FIJI installed in your macOS system")

        command = (
            "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
            + " --headless --console -macro '"
            + TEMP_THRESH
            + "' "
            + inputimage
        )
    else:
        if (not os.path.isdir(os.path.expanduser("~/Fiji.app"))):
            print("[TRONOS] [ERROR] Could not find FIJI installed in your Linux system")

        command = (
            "~/Fiji.app/ImageJ-linux64"
            + " --headless --console -macro '"
            + TEMP_THRESH
            + "' "
            + inputimage
        )

    try:
        p = subprocess.Popen(
            command, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
        p.wait()
    except Exception:
        print("[TRONOS] [ERROR] Could not execute image thresholding")


# Make parser object
def cmdline_args():
    p = argparse.ArgumentParser(
        description="""
        Tronos allows reconstruction of variable size trajectories in .dv files, automatically.
        Tools such as track-labeling and neural allow annotation of blobs according to features
        such as cellular status or brightfield morphology.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--source", nargs="+", help="Source DV file to regenerate")
    p.add_argument("--workdir", nargs="+", help="Destination directory")
    p.add_argument(
        "-p", "--project", action="store_true", help="Project the z-stack in a new file"
    )
    p.add_argument(
        "--convert", action="store_true", help="Enables conversion from dv to tiff"
    )
    p.add_argument(
        "-t", "--track", action="store_true", help="Track particle(s) route(s)"
    )
    p.add_argument(
        "--reference",
        action="store_true",
        help="Process reference image",
        default=False,
    )
    p.add_argument(
        "--first",
        action="store_true",
        help="Process only first reference image",
        default=False,
    )
    p.add_argument(
        "--sizes",
        nargs=2,
        type=int,
        help="Minimum and maximum particle size in px",
        default=[500, 10000],
    )
    p.add_argument(
        "--minframes", type=int, help="Tracking minimum frames present", default=170
    )
    p.add_argument(
        "--maxdist", type=int, help="Max. distance (pixels) that particles move between frames", default=120
    )
    p.add_argument(
        "--nullframes", type=int, help="Tracking null linking (void frames)", default=4
    )
    p.add_argument(
        "--threshold",
        type=str,
        help="Enables thresholding of specified type",
        choices=["Otsu", "MaxEntropy", "Yen", "RenyiEntropy"],
    )
    p.add_argument(
        "--blur",
        action="store_true",
        help="Generate Gaussian Blur as intermediate step; recommended for low S/N ratios.",
        default=False,
    )
    p.add_argument(
        "--crop",
        nargs=4,
        help="Crop the stack between the coordinates specified as X1 Y1 X2 Y2",
        default=[],
    )
    p.add_argument(
        "-i",
        "--individual",
        action="store_true",
        help="Save destination images as separate files",
    )
    p.add_argument(
        "-n",
        "--number",
        nargs=1,
        help="""Number of images to process per file.
        Be cautious, if file size exceeded, a thread exception will occur.""",
    )
    p.add_argument(
        "--spinning",
        nargs="+",
        type=str,
        help="Folder names for each channel in a SpinningDisk type file structure",
    )

    return p.parse_args()


# Dictionary for opening functions
opener = {".tif": open_tif, ".tiff": open_tif, ".dv": open_dv}

# Dictionary for saving functions
saveformat = {
    ".tif": tifffile.imsave,
    ".TIF": tifffile.imsave,
    ".tiff": tifffile.imsave,
    ".TIFF": tifffile.imsave,
    ".dv": mrc.imsave,
    ".DV": mrc.imsave,
}


if __name__ == "__main__":
    print(header)
    print("Welcome to TRONOS {} by {}".format(__version__, __author__))
    print("[INFO] Parsing arguments...")
    args = cmdline_args()
    if args.source:
        for fname in args.source:

            # Support SpinningDisk file type
            if os.path.isdir(fname):
                channels = args.spinning
                print("[INFO] Reconstructing SpinningDisk images to single tiff")
                fname = join_spinning(fname, channels)

            im = TronosImage(fname)
            outname = os.path.basename(os.path.splitext(fname)[0])

            fmt = str.lower(os.path.splitext(fname)[-1])
            if args.convert:
                fmt = ".tif"

            if args.reference:
                f, p = read_header(fname)
                p.outname = os.path.basename(os.path.splitext(fname)[0])
                reference_convert(f, p)
                continue

            if args.project:
                print("[INFO] Z-Stacking will occur now")
                im = max_projection(im)
                print("[INFO] Saving projected image to {}".format(fmt))
            else:
                im = (im.image).astype(np.uint8)

            imName = args.workdir[0] + "/{}{}".format(outname, fmt)
            saveformat[fmt](
                imName, im,
            )

            if args.threshold is not None:
                print(
                    "[INFO] Generating and saving {} threshold image".format(
                        args.threshold
                    )
                )
                thresholdName = args.workdir[0] + "/{}_PRJ".format(outname)
                threshold(
                    imName,
                    thresholdName + ".tif",
                    method=args.threshold,
                    blur=args.blur,
                    crop=args.crop
                )

            if args.track:
                tracker.detect_and_track(
                    thresholdName, args.maxdist, args.minframes, args.nullframes, args.sizes
                )

            if args.reference:
                reference_convert(im, outname)

            print("[INFO] Completed processing of {}".format(outname))

    print("[INFO] Program execution finished!")
    exit(0)

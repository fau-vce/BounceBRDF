import sys
import logging

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("BounceBRDF - extract_envmap.py")

import scipy
import OpenEXR
import multiprocessing
import tqdm
import numpy as np
from pathlib import Path
from functools import partial
from matplotlib import pyplot as plt
import matplotlib.backend_bases
import colour
import signal
import argparse  
import glob

SAVE_INTERMEDIATE_IMAGES = False

def save_exr(path, image_data):
    channels = {
        "RGB": np.astype(image_data, np.float16)
    }
    header = {
        "compression": OpenEXR.PIZ_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(str(path))

def ui_crop_sphere(image, *, ui_exposure=0.0):
    """Opens the user interface (UI) for selecting a sphere and returns its center and radius"""
    circle_points = []
    circle_center = None
    circle_radius = np.inf

    fig, ax = plt.subplots(tight_layout=True)
    fig.canvas.manager.set_window_title("BounceBRDF - Select mirror sphere")

    def update_title():
        if len(circle_points) < 3:
            title = f"Select 3 points on a circle around the mirror sphere"
        else:
            title = f"Selection done!\nClose the window to continue, or click again to restart"
        fig.suptitle(title)
    
    update_title()

    def find_circle(p1, p2, p3):
        """
        Input: 3 points on a circle.
        Output: center, radius of a circle defined by the 3 points.
        """
        p2_squared = p2[0] * p2[0] + p2[1] * p2[1]
        p2_to_p1 = (p1[0] * p1[0] + p1[1] * p1[1] - p2_squared) / 2
        p3_to_p2 = (p2_squared - p3[0] * p3[0] - p3[1] * p3[1]) / 2
        det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
        
        if np.isclose(abs(det), 0.0):
            return (None, np.inf)
        
        cx = (p2_to_p1*(p2[1] - p3[1]) - p3_to_p2*(p1[1] - p2[1])) / det
        cy = ((p1[0] - p2[0]) * p3_to_p2 - (p2[0] - p3[0]) * p2_to_p1) / det        
        radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

        return ((cx, cy), radius)

    def on_click(event):
        nonlocal circle_points, circle_center, circle_radius
        if event.button is matplotlib.backend_bases.MouseButton.RIGHT:
            circle_points.append((event.xdata, event.ydata))
            ax.scatter(event.xdata, event.ydata, marker="x", color="C1")
            update_title()
            if len(circle_points) == 3:
                (cx, cy), radius = find_circle(*circle_points)
                circle_center = (cx, cy)
                circle_radius = radius
                logging.debug(f"Selected circle points: {circle_points}")
                logging.debug(f"Computed circle center: {circle_center}, radius {circle_radius}")
                c = plt.Circle((cx, cy), radius, fill=False, lw=2, ec='C1')
                [p.remove() for p in reversed(ax.patches)]
                [ch.remove() for ch in reversed(ax.get_children()) if type(ch) is matplotlib.collections.PathCollection]
                ax.add_patch(c)
                ax.scatter(cx, cy, marker="x", color='C1', s=50)
                circle_points.clear()
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    ax.imshow(np.clip(colour.cctf_encoding(image * (2.0**ui_exposure)), 0.0, 1.0))
    ax.set_title("Use the right mouse button for the selection\nUse the toolbar and left mouse button to zoom or pan", fontsize=10)
    plt.show(block=True)
    return circle_center, circle_radius

def get_envmap_from_image(image, sphere_center, sphere_radius, resolution=None):
    """
    Input: image and coordinates of the mirror sphere in the image.
    Output: normalized environment map and brightness normalization factor (ensuring the upper hemisphere integrates to 1.0).
    """
    sphere_crop = image[int(sphere_center[1]-sphere_radius):int(sphere_center[1]+sphere_radius),int(sphere_center[0]-sphere_radius):int(sphere_center[0]+sphere_radius)]
    
    if resolution is None:
        resolution = 2*int(sphere_radius)
    sphere_uv_all = np.zeros((resolution,2*resolution,2), dtype=np.float32)
    r_all = np.zeros((resolution,2*resolution,3), dtype=np.float32)
    for x in range(sphere_uv_all.shape[0]):
        for y in range(sphere_uv_all.shape[1]):
            uv = np.array([x / sphere_uv_all.shape[0] * np.pi, y / sphere_uv_all.shape[1] * 2 * np.pi])
            r = np.array([np.sin(uv[0]) * np.sin(uv[1]), np.cos(uv[0]), -np.sin(uv[0]) * np.cos(uv[1])])
            sphere_uv = 1.0 / np.sqrt(2.0 * (r[1] + 1.0)) * np.array([-r[2], -r[0]])
            if np.any(np.isnan(sphere_uv)):
                continue
            sphere_uv = (sphere_uv + 1.0) / 2.0 * sphere_crop.shape[0:2]
            sphere_uv = np.clip(sphere_uv, 0, sphere_crop.shape[0])
            sphere_uv_all[x,y] = sphere_uv
            r_all[x,y] = r

    projected = np.zeros((*sphere_uv_all.shape[0:2], 3))
    for channel in range(3):
        projected[:,:,channel] = scipy.ndimage.map_coordinates(sphere_crop[:,:,channel], sphere_uv_all.reshape(-1,2).T).reshape(projected[...,channel].shape)

    return projected

def process_image(path, *, sphere_center, sphere_radius):
    """
    Input: image path and coordinates of the mirror sphere in the image.
    Saves the extracted environment map to `*.envmap.exr` and the image with normalized brightness to `*.normalized.exr`.
    """
    logging.debug(path)
    image = OpenEXR.File(str(path)).channels()["RGB"].pixels.astype(np.float32)

    envmap = get_envmap_from_image(image, sphere_center, sphere_radius, resolution=512)
    save_exr(path.parent / (path.stem + ".envmap.exr"), envmap)

def pool_raise_on_interrupt():
    """Python gets stuck if you Ctrl+C while a Pool is running (an infinite loop of KeyboardInterrupt)
       This is a workaround that converts SIGINT into InterruptedError that can be caught outside the Pool"""
    def interrupted(_1,_2):
        raise InterruptedError
    signal.signal(signal.SIGINT, interrupted)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more file paths and/or glob patterns, e.g. 'images/*.exr' or images/1.exr images/2.exr; ignores all existing *.envmap.* files unless --no-skip is set",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        default=False,
        help="Do not skip *.envmap.* files; process them alongside other inputs",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        metavar="N",
        help="Number of worker threads/processes (default: 8; use 1 for single-threaded)",
    )
    parser.add_argument(
        "--ui-exposure",
        type=float,
        default=0.0,
        metavar="EV",
        help="Exposure in stops (2^EV) to make the UI preview brighter or darker; does not affect processed output (default: 0.0)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", default=False,
        help="Allow overwriting existing output files (*.envmap.exr); without this flag the script will stop if any of these files already exist",
    )
    args = parser.parse_args()

    files = []
    seen = set()
    skipped = set()
    for pattern in args.files:
        for m in glob.glob(pattern):
            path = Path(m)
            if not args.no_skip and ".envmap." in path.name:
                skipped.add(path)
                continue
            if path not in seen:
                seen.add(path)
                files.append(path)

    if not files:
        parser.error(f"No files matched: {args.files}")

    existing = [path.parent / (path.stem + ".envmap.exr") for path in files if (path.parent / (path.stem + ".envmap.exr")).exists()]
    if existing and not args.force:
        parser.error(
            "The following output files already exist:\n"
            + "\n".join(f"  {f}" for f in existing)
            + "\nRun the script again with the --force (or -f) argument to allow overwriting existing files."
        )

    if len(skipped) > 0:
        logging.info(f"Skipped {len(skipped)} files: {[str(file) for file in skipped]}")
    logging.info(f"Loading {len(files)} files: {[str(file) for file in files]}")
    logging.info(f"Using the first file as template: {str(files[0])}")

    image = OpenEXR.File(str(files[0])).channels()["RGB"].pixels.astype(np.float32)

    logging.info(f"Opening UI for sphere cropping (exposure 2^({args.ui_exposure}))")
    sphere_center, sphere_radius = ui_crop_sphere(image, ui_exposure=args.ui_exposure)
    if sphere_center is None:
        raise InterruptedError("Sphere selection interrupted without 3 points being selected")
    logging.info(f"Selected the following: center coordinates: {sphere_center} px | radius: {sphere_radius} px")

    # From now on, all images are processed independently, in parallel:
    process_image_with_partial_arguments = partial(
        process_image,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius
    )

    logging.info(f"Starting batch processing with {args.threads} thread(s)")
    with multiprocessing.Pool(processes=args.threads, initializer=pool_raise_on_interrupt) as pool:
        try:
            with tqdm.tqdm(total=len(files)) as progress_bar:
                for return_value in pool.imap_unordered(process_image_with_partial_arguments, files):
                    progress_bar.update(1)
        except Exception as e:
            logging.warning("Execution interrupted")
            logging.error(e)
            pool.terminate()
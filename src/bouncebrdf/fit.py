import sys
import logging

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("BounceBRDF - fit.py")
    logging.info("Starting this script might take a while as it loads PyTorch and Mitsuba")

import argparse
import numpy as np
import mitsuba as mi
import tqdm
from pathlib import Path
import skimage.transform
from timeit import default_timer as timer
import glob

from bouncebrdf.render import *
from bouncebrdf.neuralnet import *

def save_renderer_textures(renderer, output_path):
    textures = {
        TextureType.NORMAL_RGB: renderer._scene_params["surface.bsdf.normalmap.data"].numpy(),
        TextureType.BASECOLOR: renderer._scene_params["surface.bsdf.nested_bsdf.base_color.data"].numpy(),
        TextureType.ROUGHNESS: renderer._scene_params["surface.bsdf.nested_bsdf.roughness.data"].numpy()[...,0],
        TextureType.METALLIC: renderer._scene_params["surface.bsdf.nested_bsdf.metallic.data"].numpy()[...,0],
    }
    save_textures(output_path, textures)

def integrate_envmap_hemisphere(envmap):
    """Computes the integral over the upper hemisphere from an environment map"""
    sin_theta = np.sin(np.linspace(0.0, np.pi, envmap.shape[0], endpoint=False))[:,None,None]
    cos_theta = np.cos(np.linspace(0.0, np.pi, envmap.shape[0], endpoint=False))[:,None,None]
    sin_cos_envmap = np.clip(sin_theta * cos_theta * envmap, 0.0, None)
    return 2.0 * np.pi * np.mean(sin_cos_envmap, axis=(0,1))

def load_references(paths, crop_fn, img_scale, eval=False):
    references = []
    reference_resolution = None
    for path in tqdm.tqdm(paths, desc="Loading input images"):
        reference = {}
        reference["name"] = path.stem
        reference["evaluation_only"] = eval
        reference["image"] = mi.Bitmap(str(path))
        if ".registered." in path.name:
            reference["environment_map"] = mi.Bitmap(str(path.parent / path.name.replace(".registered.", ".envmap.")))
        elif ".normalized." in path.name:
            reference["environment_map"] = mi.Bitmap(str(path.parent / path.name.replace(".normalized.", ".envmap.")))
        elif "blender" in path.parent.name:
            blender_path = Path("../envmaps/blender") / path.name.replace(".exr", ".envmap.exr")
            reference["environment_map"] = mi.Bitmap(str(blender_path))
        elif (path.parent / path.name.replace(".exr", ".envmap.exr")).exists():
            reference["environment_map"] = mi.Bitmap(str(path.parent / path.name.replace(".exr", ".envmap.exr")))
        else:
            raise LookupError("Could not find the environment map file")
        
        reference["brightness_normalization_factor"] = 1.0 / integrate_envmap_hemisphere(np.array(reference["environment_map"]))

        res = (reference["image"].height(), reference["image"].width())
        if reference_resolution is None:
            reference_resolution = res
        assert res == reference_resolution, "All input images must have the same resolution"
        reference["image_downsampled"] = skimage.transform.rescale(crop_fn(np.array(reference["image"], dtype=np.float32)), img_scale, anti_aliasing=False, channel_axis=2)
        references.append(reference)
    return references

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Fit SVBRDF textures to multi-illumination photos using neural prediction."
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more file paths and/or glob patterns, e.g. 'images/*.exr' or images/1.exr images/2.exr; a matching *.envmap.exr must exist for each input image; run extract_envmap.py first to generate them",
    )
    parser.add_argument(
        "--subset", type=int, default=-1, metavar="N",
        help="Use only the first N files sorted alphabetically; -1 means all files (default: -1)",
    )
    parser.add_argument(
        "--crop", nargs=4, type=int, metavar=("TOP", "LEFT", "HEIGHT", "WIDTH"), default=None,
        help="Crop input images before processing, specified as top-left corner and dimensions (default: no crop)",
    )
    parser.add_argument(
        "--resolution-scaling", type=float, default=1.0,
        help="Scale factor applied to input images after cropping (1.0 = original resolution, 0.5 = half size; default: 1.0)",
    )
    parser.add_argument(
        "--training-iterations", type=int, default=1000,
        help="Number of training iterations (default: 1000)",
    )
    parser.add_argument(
        "--training-batch-size", type=int, default=4096,
        help="Training batch size; total samples = batch_size * iterations (default: 4096)",
    )
    parser.add_argument(
        "--training-noise-scale", type=float, default=0.1,
        help="Noise scale applied during training (default: 0.1)",
    )
    parser.add_argument(
        "--training-learning-rate", type=float, default=0.001,
        help="Learning rate during training (default: 0.001)",
    )
    parser.add_argument(
        "--load-trained-weights", type=Path, default=None, metavar="WEIGHTS_FILE",
        help="Skip training and load model weights from this file instead",
    )
    parser.add_argument(
        "--mitsuba-variant", default="cuda_ad_rgb" if device == "cuda" else "llvm_ad_rgb",
        help="Mitsuba variant to use (default: cuda_ad_rgb if CUDA is available, else llvm_ad_rgb - either CUDA or LLVM must be installed)",
    )
    parser.add_argument(
        "--no-log", action="store_true", default=False,
        help="Disable log-space reflectance preconditioning in the neural network; useful if the photos are very noisy (default: log enabled)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", default=False,
        help="Allow overwriting existing output files (trained_model_weights.pth, basecolor.png, metallic.png, normal.png, roughness.png); without this flag the script will stop if any of these files already exist",
    )
    args = parser.parse_args()

    mi.set_variant(args.mitsuba_variant)
    logging.info(f"Using '{mi.variant()}' as variant for Mitsuba")

    def expand_globs(patterns):
        result = []
        seen = set()
        for pattern in patterns:
            for m in glob.glob(pattern):
                path = Path(m)
                if path not in seen and ".envmap." not in path.name:
                    seen.add(path)
                    result.append(path)
        return sorted(result, key=lambda p: p.name)

    files = expand_globs(args.files)
    if not files:
        parser.error(f"No files matched: {args.files}")

    if args.subset != -1:
        if args.subset > len(files):
            logging.warning(f"--subset {args.subset} is larger than the number of matched files ({len(files)}); using all files")
        else:
            logging.info(f"Skipping all files above {args.subset} (--subset)")
            files = files[:args.subset]

    training_iterations = args.training_iterations
    training_noise_scale = args.training_noise_scale
    training_batch_size = args.training_batch_size
    training_samples = training_batch_size * training_iterations

    if args.crop is not None:
        t, l, h, w = args.crop
        crop_fn = lambda arr: arr[t:t+h, l:l+w]
    else:
        crop_fn = lambda arr: arr

    output_path = files[0].parent / f"results_{training_iterations}_{training_noise_scale}"

    OUTPUT_FILES = ["trained_model_weights.pth", "basecolor.png", "metallic.png", "normal.png", "roughness.png"]
    existing = [f for f in OUTPUT_FILES if (output_path / f).exists()]
    if existing and not args.force:
        parser.error(
            f"The following output files already exist in '{output_path}':\n"
            + "\n".join(f"  {f}" for f in existing)
            + "\nRun the script again with the --force (or -f) argument to allow overwriting existing files."
        )

    logging.info(f"Loading {len(files)} files: {[str(f) for f in files]}")

    output_path.mkdir(exist_ok=True)

    references = load_references(files, crop_fn=crop_fn, img_scale=args.resolution_scaling)
    logging.info(f"Loading finished with {len(references)} images")

    image_resolution = None
    if len(references) > 0:
        image_resolution = references[0]["image_downsampled"].shape[0:2]

    logging.info("Preparing the predictive neural network")

    training_envmaps = np.array([
        reference["brightness_normalization_factor"]*np.array(reference["environment_map"]) for reference in references if not reference["evaluation_only"]
    ])

    neural_predictor = NeuralBRDFPredictor(len(training_envmaps), apply_log=not args.no_log, noise=training_noise_scale).to(device)
    mem, mem_params, mem_bufs = model_memory(neural_predictor)
    logging.info(f"Model memory usage: {(mem / 1024 / 1024):.3f} MB (out of which {(mem_bufs / 1024 / 1024):.3f} MB are buffers)")

    if args.load_trained_weights is not None:
        logging.info(f"Training skipped, loading the saved model state from {args.load_trained_weights}")
        neural_predictor.load_state_dict(torch.load(args.load_trained_weights, weights_only=True))
    else:
        neural_trainer = Trainer(model=neural_predictor, envmaps=training_envmaps, batch_size=training_batch_size, lr=args.training_learning_rate, mitsuba_variant=mi.variant())
        training_start_time = timer()
        neural_trainer.train(samples=training_samples)
        training_end_time = timer()
        logging.info(f"Saving trained model weights to {str(output_path / 'trained_model_weights.pth')}")
        torch.save(neural_predictor.state_dict(), output_path / "trained_model_weights.pth")
        logging.info(f"Training time: {(training_end_time-training_start_time):.1f} seconds")

    logging.info("Predicting svBRDF using the neural network")
    reflectances = torch.zeros((np.prod(image_resolution), 3, len(training_envmaps)), device=device)
    for i, reference in enumerate([reference for reference in references if not reference["evaluation_only"]]):
        reflectances[:,:,i] = torch.reshape(
            torch.tensor(reference["brightness_normalization_factor"] * reference["image_downsampled"], device=device),
            (-1,3)).to(device)

    neural_predictor.eval()
    inference_start_time = timer()
    predicted_textures_torch = split_texture_parameters(neural_predictor.forward_chunked(reflectances, max_chunk=training_batch_size))
    inference_end_time = timer()
    logging.info(f"Inference finished in {(inference_end_time-inference_start_time):.3f} seconds")
    predicted_textures = {}
    for key in predicted_textures_torch.keys():
        predicted_textures[key] = np.reshape(predicted_textures_torch[key].cpu().numpy().copy(), (*image_resolution, -1))
    del predicted_textures_torch
    if device == "cuda":
        torch.cuda.empty_cache()

    optimization_resolution = predicted_textures[TextureType.BASECOLOR].shape[0:2]
    renderer = MaterialSampleRenderer(resolution=optimization_resolution, mitsuba_variant=mi.variant())
    renderer.set_textures(predicted_textures)
    logging.info(f"Saving the predicted textures to {str(output_path)}")
    save_renderer_textures(renderer, output_path=output_path)
    logging.info("Finished")
    
    mi.set_log_level(mi.LogLevel.Error)  # suppress harmless "Destructor called while thread 'main' was still running" at shutdown

import sys
import logging

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("BounceBRDF - render.py")
    logging.info("Starting this script might take a while as it loads Mitsuba")

from pathlib import Path
import numpy as np
import mitsuba as mi

from bouncebrdf.utils import *

class MaterialSampleRenderer:

    def __init__(self, resolution, mitsuba_variant="cuda_ad_rgb"):
        mi.set_variant(mitsuba_variant)
        self._resolution = resolution
        self._scene = self._create_scene(resolution)
        self._scene_params = mi.traverse(self._scene)

    def set_textures(self, material_textures):
        def reshape(input):
            return input.reshape((self._resolution[0], self._resolution[1], -1))
        
        def to_numpy(x):
            if isinstance(x, np.ndarray):
                return x
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

        if TextureType.NORMAL_RGB in material_textures:
            self._scene_params["surface.bsdf.normalmap.data"] = to_numpy(reshape(material_textures[TextureType.NORMAL_RGB]))
        elif TextureType.NORMAL_XY in material_textures:
            normal_rgb = vector_to_normal_map_rgb(xy_to_dir(to_numpy(reshape(material_textures[TextureType.NORMAL_XY]))))
            self._scene_params["surface.bsdf.normalmap.data"] = normal_rgb
        
        self._scene_params["surface.bsdf.nested_bsdf.base_color.data"] = to_numpy(reshape(material_textures[TextureType.BASECOLOR]))
        self._scene_params["surface.bsdf.nested_bsdf.metallic.data"] = to_numpy(reshape(material_textures[TextureType.METALLIC][...,0:1]))
        self._scene_params["surface.bsdf.nested_bsdf.roughness.data"] = to_numpy(reshape(material_textures[TextureType.ROUGHNESS][...,0:1]))
        
        self._scene_params.update()

    def set_environment_map(self, envmap = None, scale = None):
        if envmap is not None:
            self._scene_params["envmap.data"] = np.array(envmap).copy()
        if scale is not None:
            self._scene_params["envmap.scale"] = scale
        self._scene_params.update()

    def render(self, spp):
        return mi.render(self._scene, self._scene_params, spp=spp, sensor=0)

    def _create_disney_material(self, texture_size):
        dummy_texture_rgb = np.zeros((texture_size[0],texture_size[1],3))
        dummy_texture_mono = np.zeros((texture_size[0],texture_size[1],1))

        return {
            "type": "normalmap",
            "normalmap": {
                "type": "bitmap",
                "raw": True, # True -> no gamma mapping
                "filter_type": "nearest",
                "data": dummy_texture_rgb
            },
            "specular": {
                "type": "principled",
                "base_color": {
                    "type": "bitmap",
                    "raw": True, # True -> no gamma mapping
                    "filter_type": "nearest",
                    "data": dummy_texture_rgb
                },
                "metallic": {
                    "type": "bitmap",
                    "raw": True, # True -> no gamma mapping
                    "filter_type": "nearest",
                    "data": dummy_texture_mono
                },
                "roughness": {
                    "type": "bitmap",
                    "raw": True, # True -> no gamma mapping
                    "filter_type": "nearest",
                    "data": dummy_texture_mono
                },
            },
        }

    def _create_scene(self, texture_size):
        film_width = texture_size[1]
        film_height = texture_size[0]
        width_to_height_aspect = film_width / film_height
        rectangle_halfwidth = 1.0 / 2.0 * 10
        rectangle_halfheight = 1.0 / width_to_height_aspect / 2.0 * 10

        material = self._create_disney_material(texture_size)

        scene = mi.load_dict({
            "type": "scene",
            "envmap": {
                "type": "envmap",
                "bitmap": mi.Bitmap(np.ones((10,5))),
                "mis_compensation": False,
                "scale": 1.0,
                "sampling_weight": 1.0,
            },
            "surface": {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f().translate([0,0,0]).scale([-rectangle_halfwidth,1,rectangle_halfheight]).rotate([1,0,0],-90),
                "material": material,
            },
            "orthographic_camera": {
                "type": "orthographic",
                "to_world": mi.ScalarTransform4f().look_at(
                    origin=[0,1,0],
                    target=[0,0,0],
                    up=[0,0,1]
                ).scale([max(rectangle_halfwidth,rectangle_halfheight),max(rectangle_halfwidth,rectangle_halfheight),1]),
                "film": {
                    "type": "hdrfilm",
                    "width": film_width,
                    "height": film_height,
                    "rfilter": {"type": "box"},
                }
            },
            "integrator": {
                "type": "prb",
                "max_depth": 2,
            }
        })

        return scene

if __name__ == '__main__':
    import argparse
    import glob
    import colour
    from PIL import Image

    parser = argparse.ArgumentParser(
        description="Render a material with given Disney SVBRDF textures using the provided environment maps as illumination."
    )
    parser.add_argument(
        "output", type=Path, default=None, metavar="OUTPUT_DIRECTORY",
        help="Directory where all rendered images will be saved",
    )
    parser.add_argument(
        "textures", type=Path, default=None, metavar="TEXTURES_DIRECTORY",
        help=f"Directory containing {', '.join(t.filename + '.png' for t in [TextureType.NORMAL_RGB, TextureType.BASECOLOR, TextureType.METALLIC, TextureType.ROUGHNESS])} textures",
    )
    parser.add_argument(
        "envmaps",
        nargs="+",
        metavar="ENVMAP_FILE",
        default=[],
        help="Environment maps in .exr format that are used to render the material; one render per each environment map",
    )
    parser.add_argument(
        "--spp", type=int, default=64,
        help="Samples per pixel; low spp results in noisy renders (default: 64)",
    )
    parser.add_argument(
        "--mitsuba-variant", default="cuda_ad_rgb",
        help="Mitsuba variant to use (default: cuda_ad_rgb if CUDA is available, else llvm_ad_rgb - either CUDA or LLVM must be installed)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", default=False,
        help="Allow overwriting existing output files; without this flag the script will stop if any output file already exists",
    )
    args = parser.parse_args()

    def expand_globs(patterns):
        result = []
        seen = set()
        for pattern in patterns:
            for m in glob.glob(pattern):
                path = Path(m)
                if path not in seen:
                    seen.add(path)
                    result.append(path)
        return sorted(result, key=lambda p: p.name)

    envmap_paths = expand_globs(args.envmaps)
    if not envmap_paths:
        parser.error(f"No envmap files matched: {args.envmaps}")

    args.output.mkdir(parents=True, exist_ok=True)

    output_paths = [args.output / (p.stem + ".render.exr") for p in envmap_paths]
    existing = [str(p) for p in output_paths if p.exists()]
    if existing and not args.force:
        parser.error(
            f"The following output files already exist in '{args.output}':\n"
            + "\n".join(f"  {f}" for f in existing)
            + "\nRun the script again with --force (or -f) to allow overwriting existing files."
        )

    textures_path = args.textures
    textures = {}
    textures[TextureType.NORMAL_RGB] = np.array(Image.open(textures_path / f"{TextureType.NORMAL_RGB.filename}.png")) / 255
    textures[TextureType.ROUGHNESS] = (np.array(Image.open(textures_path / f"{TextureType.ROUGHNESS.filename}.png")) / 255)[..., None]
    textures[TextureType.METALLIC] = (np.array(Image.open(textures_path / f"{TextureType.METALLIC.filename}.png")) / 255)[..., None]
    textures[TextureType.BASECOLOR] = colour.cctf_decoding(np.array(Image.open(textures_path / f"{TextureType.BASECOLOR.filename}.png")) / 255)

    resolution = textures[TextureType.NORMAL_RGB].shape[0:2]

    mi.set_variant(args.mitsuba_variant)
    logging.info(f"Using '{mi.variant()}' as variant for Mitsuba")

    renderer = MaterialSampleRenderer(resolution=resolution, mitsuba_variant=args.mitsuba_variant)
    renderer.set_textures(textures)

    logging.info(f"Loaded textures at resolution {resolution} from '{textures_path}'")
    logging.info(f"Rendering {len(envmap_paths)} envmap(s) with spp={args.spp}")

    for envmap_path, output_path in zip(envmap_paths, output_paths):
        logging.info(f"Rendering '{envmap_path.name}' -> '{output_path.name}'")
        envmap = mi.Bitmap(str(envmap_path))
        renderer.set_environment_map(envmap)
        render = renderer.render(spp=args.spp)
        mi.Bitmap(render).write(str(output_path))

    logging.info("Finished")

    mi.set_log_level(mi.LogLevel.Error)  # suppress harmless "Destructor called while thread 'main' was still running" at shutdown

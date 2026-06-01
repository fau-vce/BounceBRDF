import numpy as np
import torch
import colour
from enum import Enum, auto
from PIL import Image

class TextureType(Enum):

    NORMAL_RGB = auto()
    """Normal map encoded in RGB"""

    NORMAL_XY = auto()
    """Normal map in x,y coordinates normalized to z=1"""

    ROUGHNESS = auto()
    BASECOLOR = auto()
    METALLIC = auto()

    @property
    def filename(self):
        _names = {
            TextureType.NORMAL_RGB: "normal",
            TextureType.NORMAL_XY: "normal_xy",
        }
        return _names.get(self, self.name.lower())

def normalized(x):
    if torch.is_tensor(x):
        return x / torch.linalg.norm(x, keepdim=True, dim=-1)
    else:
        return x / np.linalg.norm(x, keepdims=True, axis=-1)

def vector_to_normal_map_rgb(vector):
    """Encodes vectors (xyz) into normal map pixels (rgb)"""
    return (normalized(vector) + 1.0) / 2.0

def xy_to_dir(xy):
    """Converts from a xy normal (z=1) to a normalized direction (xyz)
    """
    if torch.is_tensor(xy):
        xyz = torch.ones((*xy.shape[:-1], 3), device=xy.device)
    else:
        xyz = np.ones((*xy.shape[:-1], 3))
    xyz[..., :2] = xy
    xyz = normalized(xyz)
    return xyz

def save_textures(out_dir, textures):
    out_dir.mkdir(exist_ok=True)
    for texture_type, texture in textures.items():
        texture_to_save = texture
        if texture_type == TextureType.BASECOLOR:
            texture_to_save = colour.cctf_encoding(texture_to_save)
        texture_8bit = np.astype(np.clip(texture_to_save, 0, 1) * 255.0, np.uint8)
        Image.fromarray(texture_8bit).save(out_dir / f"{texture_type.filename}.png")

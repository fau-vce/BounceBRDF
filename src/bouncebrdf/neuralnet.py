import torch
from torch import nn
import tqdm
import logging

from .render import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logging.info(f"Using '{device}' device for PyTorch")

def model_memory(model: nn.Module):
    """Computes how much memory a given `torch.nn.Module` takes"""
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes
    return mem, mem_params, mem_bufs

class MLPBackbone(nn.Module):
    def __init__(self, c_in, token_cnt, c_out, nf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * token_cnt, nf),
            nn.ReLU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.ReLU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.ReLU(),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, c_out),
        )

    def forward(self, x):
        return self.net(x)

class NeuralBRDFPredictor(nn.Module):
    def __init__(self, envmaps_count, nf=2**10, apply_log=True, noise=0.0):
        super().__init__()
        self.apply_log = apply_log
        self.net = MLPBackbone(3, envmaps_count, 7, nf)

        self.register_buffer('bias', torch.tensor([-0.5, -0.5, 0, 0, 0, 0, 0]), persistent=False)
        self.register_buffer('scale', torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]), persistent=False)
        self.noise_scale = noise

    def augment(self, x, p=0.85, log_scale=False):
        assert x.ndim == 3
        if not self.training:
            return x
        if log_scale:
            noise = torch.exp(torch.randn_like(x) * self.noise_scale)
        else:
            noise = torch.randn_like(x) * self.noise_scale
        mask = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > p  # Mask the channels c together

        if log_scale:
            noise.masked_fill_(mask.expand_as(noise), 1)
            return x * noise
        else:    
            noise.masked_fill_(mask.expand_as(noise), 0)
            return x + noise

    def precondition(self, reflectances):
        if self.apply_log:
            # Reflectances tend to be [0.0001, 9] = [e^(-9), e^2]
            return torch.log(torch.clip(reflectances, 1e-6, 100)) / 5 + 0.7
        else:
            return reflectances * 2 - 1  # ~[-1, 1] is a better range for NN

    def forward(self, x):
        x = self.augment(x)
        x = self.precondition(x)  # Compute rations after adding noise to the observations
        predicted_params = torch.sigmoid(self.net(x))

        # XY coordinates from [0, 1] to [-0.5, 0.5] to (-1.0,1.0), which roughly clips normal vectors to X degrees polar angle
        predicted_params = (predicted_params + self.bias) * self.scale
        return predicted_params

    @torch.no_grad()
    def forward_chunked(self, x, max_chunk=2 ** 12):
        """
        This method is equivalent to `forward` but it chunks the input to preserve GPU memory.
        It is especially useful for processing high-resolution textures.
        """
        assert not self.training
        chunks = (x.shape[0] - 1) // max_chunk + 1
        x_chunked = torch.chunk(x, chunks=chunks, dim=0)
        y = []
        for x_chunk in tqdm.tqdm(x_chunked):
            y_ = self(x_chunk)
            assert y_.shape[0] == x_chunk.shape[0]
            y.append(y_)
        return torch.concatenate(y, dim=0)
    
class Trainer:
    """Trains the `NeuralBRDFPredictor` on random samples rendered with the `MaterialSampleRenderer`"""

    def __init__(self, *, model: NeuralBRDFPredictor, envmaps, batch_size=4096, lr=0.001, mitsuba_variant="cuda_ad_rgb"):
        """
        - `envmaps` has shape `(number_of_envmaps, height, width, channels)`
        """
        assert batch_size % 2 == 0, "`batch_size` needs to be divisible by 2"
        self._model = model
        self._envmaps = envmaps
        self._batch_size = batch_size
        self._renderer = MaterialSampleRenderer(resolution=(2,batch_size//2), mitsuba_variant=mitsuba_variant)
        self.reset_optimizer(lr=lr)

    def reset_optimizer(self, optimizer_class=torch.optim.Adam, **params):
        self._optimizer = optimizer_class(self._model.parameters(), **params)
    
    def train(self, samples, spp=1024):
        self._model.train()
        for iteration in (progress_bar := tqdm.tqdm(range(samples // self._batch_size))):
            ground_truth_params = random_sample_texture_parameters(self._batch_size, device)
            self._renderer.set_textures(ground_truth_params)
            ground_truth_reflectances = torch.zeros((self._batch_size, 3, len(self._envmaps)), device=device)
            for envmap_id, envmap in enumerate(self._envmaps):
                self._renderer.set_environment_map(envmap, scale=1.0)
                render = self._renderer.render(spp)
                ground_truth_reflectances[..., envmap_id] = render.torch().reshape(-1, 3)

            predictions = self._model(ground_truth_reflectances)
            predicted_texture_parameters = split_texture_parameters(predictions)
            losses = compute_losses(predicted_texture_parameters, ground_truth_params)
            loss = losses["total"]
            if torch.any(torch.isnan(loss)):
                logging.error("NaN detected in loss, training stopped")
                break
            loss.backward()
            progress_bar.set_description(f"Neural network training - loss: {loss.detach().cpu().numpy():.6f}")
            self._optimizer.step()
            self._optimizer.zero_grad()

def random_sample_texture_parameters(batch_size, device=device):
    # Generate normals directly in xy
    r = torch.rand(batch_size, device=device)  # careful, this sampling is biased toward *up* normals, which is usually what we want; use sqrt to remove the bias
    phi = torch.rand(batch_size, device=device) * 2 * torch.pi
    normals_xy = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=-1)
                      
    metallic = torch.rand(batch_size, 1, device=device)
    metallic[0:batch_size//4] = 0.0
    metallic[batch_size//4:batch_size//2] = 1.0

    roughness = torch.rand(batch_size, 1, device=device)
    base_color = torch.rand(batch_size, 3, device=device)

    return {
        TextureType.NORMAL_XY: normals_xy,
        TextureType.METALLIC: metallic,
        TextureType.ROUGHNESS: roughness,
        TextureType.BASECOLOR: base_color
    }

def split_texture_parameters(parameters):
    """Splits a flat parameter tensor into a dict of TextureType → tensor"""
    return {
        TextureType.NORMAL_XY: parameters[...,0:2],
        TextureType.METALLIC: parameters[...,2:3],
        TextureType.ROUGHNESS: parameters[...,3:4],
        TextureType.BASECOLOR: parameters[...,4:7],
    }

def compute_losses(predicted, target):
    loss = {}
    inner_loss_function = torch.nn.L1Loss()
    loss["normal_xy"] = inner_loss_function(predicted[TextureType.NORMAL_XY], target[TextureType.NORMAL_XY])
    loss["metallic"] = inner_loss_function(predicted[TextureType.METALLIC], target[TextureType.METALLIC])
    loss["roughness"] = inner_loss_function(predicted[TextureType.ROUGHNESS], target[TextureType.ROUGHNESS])
    loss["base_color"] = inner_loss_function(predicted[TextureType.BASECOLOR], target[TextureType.BASECOLOR])
    loss["total"] = 10.0 * loss["normal_xy"] + loss["metallic"] + loss["roughness"] + loss["base_color"]
    return loss



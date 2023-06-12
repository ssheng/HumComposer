import bentoml
import numpy as np
import torch

from audiocraft.models import MusicGen

class MusicGenRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu", "nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = MusicGen.get_pretrained('melody')
        self.model.set_generation_params(duration=16) 

    @bentoml.Runnable.method(batchable=False)
    def generate(self, description: str):
        return self.model.sample_rate, self.model.generate([description]).cpu().numpy()

    @bentoml.Runnable.method(batchable=False)
    def generate_with_chroma(self, description: str, melody: np.ndarray, sample_rate: int):
        tensor = torch.from_numpy(melody)
        return self.model.sample_rate, self.model.generate_with_chroma([description], tensor[None].expand(1, -1, -1), sample_rate).cpu().numpy()


musicgen = bentoml.Runner(MusicGenRunnable, name="musicgen")
svc = bentoml.Service("hum_composer", runners=[musicgen])

@svc.api(
    input=bentoml.io.Text.from_sample("Upbeat punk rock"),
    output=bentoml.io.NumpyNdarray(),
)
def generate(description: str) -> np.ndarray:
    return musicgen.generate.run(description)

@svc.api(
    input=bentoml.io.Multipart(
        description=bentoml.io.Text.from_sample("Upbeat punk rock"),
        melody=bentoml.io.NumpyNdarray(dtype=np.float32),
        sample_rate=bentoml.io.Text.from_sample("44100"),
    ),
    output=bentoml.io.NumpyNdarray(),
)
def generate_with_chroma(description: str, melody: np.ndarray, sample_rate: str) -> np.ndarray:
    result = musicgen.generate_with_chroma.run(description, melody, int(sample_rate))
    return result


from fastapi import FastAPI
from composer import HumComposer
from composer import create_interface

import gradio as gr

composer = HumComposer(generate, generate_with_chroma)
app = FastAPI()
app = gr.mount_gradio_app(app, create_interface(composer), path="/composer")
svc.mount_asgi_app(app, "/")

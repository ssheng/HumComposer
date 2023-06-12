import bentoml
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

model = MusicGen.get_pretrained("small")
model.set_generation_params(duration=8)  # generate 8 seconds.

print(type(model.compression_model))
print(type(model.lm))
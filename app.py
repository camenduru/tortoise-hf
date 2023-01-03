import os, sys
import tempfile
import gradio as gr
import numpy as np
from typing import Tuple, List

# Setup and installation
os.system("git clone https://github.com/neonbjb/tortoise-tts.git")
sys.path.append("./tortoise-tts/")
os.system("pip install -r ./tortoise-tts/requirements.txt")
os.system("python ./tortoise-tts/setup.py install")

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice


# Download and instantiate model
tts = TextToSpeech()

voices = ["random", "mol", "tom", "applejack", "daniel", "myself", "weaver", "train_empire", "train_dotrice", "rainbow", "pat", "geralt", "halle", "train_kennard", "jlaw", "train_grace", "angie", "william", "tim_reynolds", "train_atkins", "train_dreams", "train_mouse", "freeman", "deniro", "lj", "train_lescault", "emma", "pat2", "snakes", "train_daws"]
presets = ["ultra_fast", "fast", "standard", "high_quality"]
preset_default = "fast"

TORTOISE_SR_IN  = 22050
TORTOISE_SR_OUT = 24000


# Helper functions
def split_into_chunks(t: torch.Tensor, sample_rate, chunk_duration_sec) -> List[torch.Tensor]:
  duration = t.shape[1] / sample_rate
  num_chunks = 1 + int(duration/chunk_duration_sec)
  chunks = [t[:,(sample_rate*chunk_duration_sec*i):(sample_rate*chunk_duration_sec*(i+1))] for i in range(num_chunks)]
  return chunks

def generate_from_preset(voice: str, text, model_preset):
  voice_samples, _ = load_voice(voice)
  return tts_main(voice_samples, text, model_preset)

def generate_from_files(files: List[tempfile._TemporaryFileWrapper], text, model_preset):
  voice_samples = [load_audio(f.name, TORTOISE_SR_IN) for f in files]
  return tts_main(voice_samples, text, model_preset)

def generate_from_recording(recording, text, model_preset):
  sample_rate, audio = recording
  
  # convert to float tensor and normalize - see: https://github.com/neonbjb/tortoise-tts/blob/main/tortoise/utils/audio.py#L16
  norm_fix = 1
  if audio.dtype == np.int32:
    norm_fix = 2**31
  elif audio.dtype == np.int16:
    norm_fix = 2**15
  t = torch.FloatTensor(audio.T) / norm_fix

  # convert to mono
  if len(t.shape) > 1:
    t = torch.mean(t, axis=0).unsqueeze(0)

  # resample to 22050 hz
  t = torchaudio.transforms.Resample(sample_rate, TORTOISE_SR_IN)(t)
  
  voice_samples = split_into_chunks(t, TORTOISE_SR_IN, 10)
  return tts_main(voice_samples, text, model_preset)

def tts_main(voice_samples, text, model_preset):
  gen = tts.tts_with_preset(
    text,
    voice_samples=voice_samples,
    conditioning_latents=None,
    preset=model_preset
  )
  torchaudio.save("generated.wav", gen.squeeze(0).cpu(), TORTOISE_SR_OUT)
  return "generated.wav"
    

with gr.Blocks() as demo:
    gr.Markdown("""# TorToiSe
Tortoise is a text-to-speech model developed by James Betker. It is capable of zero-shot voice cloning from a small set of voice samples. GitHub repo: [neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts).

The Spaces implementation was created by [mdnestor](https://github.com/mdnestor). Currently in alpha; submit issues [here](https://huggingface.co/spaces/mdnestor/tortoise/discussions)!

## Usage
1. Select a voice - either by choosing a preset, uploading audio files, or recording via microphone. Recorded audio is chunked into 10 second segments. Follow the guidelines in the [voice customization guide](https://github.com/neonbjb/tortoise-tts#voice-customization-guide).
2. Choose a model preset (ultra fast/fast/standard/high quality), type the text to speak, and click **Generate**.
""")

    with gr.Row():
      # From preset voice
      with gr.Tab("Choose preset voice"):
        inp1 = gr.Dropdown(voices, value="random", label="Preset voice")
        preset = gr.Dropdown(presets, value=preset_default, label="Model preset")
        text = gr.Textbox(label="Text to speak", value="Hello, world!")
        btn1 = gr.Button("Generate")
        audio_out = gr.Audio()
        btn1.click(generate_from_preset, inputs=[inp1, text, preset], outputs=audio_out)
      
      # From file upload
      with gr.Tab("Upload audio"):
        inp2 = gr.File(file_count="multiple")
        preset = gr.Dropdown(presets, value=preset_default, label="Model preset")
        text = gr.Textbox(label="Text to speak", value="Hello, world!")
        btn2 = gr.Button("Generate")
        audio_out = gr.Audio()
        btn2.click(generate_from_files, inputs=[inp2, text, preset], outputs=audio_out)
      
      # From microphone
      with gr.Tab("Record audio"):
        inp3 = gr.Audio(source="microphone")
        preset = gr.Dropdown(presets, value=preset_default, label="Model preset")
        text = gr.Textbox(label="Text to speak", value="Hello, world!")
        btn3 = gr.Button("Generate")
        audio_out = gr.Audio()
        btn3.click(generate_from_recording, inputs=[inp3, text, preset], outputs=audio_out)
          
demo.launch()
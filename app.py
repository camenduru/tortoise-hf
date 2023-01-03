import os, sys
import tempfile
import gradio as gr
import numpy
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

voices = ["mol", "tom", "applejack", "daniel", "myself", "weaver", "train_empire", "train_dotrice", "rainbow", "pat", "geralt", "halle", "train_kennard", "jlaw", "train_grace", "angie", "william", "tim_reynolds", "train_atkins", "train_dreams", "train_mouse", "freeman", "deniro", "lj", "train_lescault", "emma", "pat2", "snakes", "train_daws"]
presets = ["ultra_fast", "fast", "standard", "high_quality"]
preset_default = "fast"

TORTOISE_SR     = 22050
TORTOISE_SR_OUT = 24000


# Helper functions
def split_into_chunks(t: torch.Tensor, sample_rate: int, chunk_duration_sec: int) -> List[torch.Tensor]:
  duration = t.shape[1] / sample_rate
  num_chunks = 1 + int(duration/chunk_duration_sec)
  chunks = [t[:,(sample_rate*chunk_duration_sec*i):(sample_rate*chunk_duration_sec*(i+1))] for i in range(num_chunks)]
  return chunks

def load_preset_voice(voice: str) -> List[torch.Tensor]:
  samples, _ = load_voice(voice)
  return samples

def load_voice_from_files(files: List[tempfile._TemporaryFileWrapper]) -> List[torch.Tensor]:
  return [load_audio(f.name, TORTOISE_SR) for f in files]

def load_voice_from_recording(recording: Tuple[int, numpy.ndarray]) -> List[torch.Tensor]:
  sample_rate, audio = recording
  with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
    torchaudio.save(temp.name, torch.Tensor(a), sample_rate)
    t = load_audio(temp.name, TORTOISE_SR)
  chunks = split_into_chunks(t, TORTOISE_SR, chunk_duration_sec=10)
  return chunks


with gr.Blocks() as demo:
    gr.Markdown("""# TorToiSe
Tortoise is a text-to-speech model developed by James Betker. It is capable of zero-shot voice cloning from a small set of voice samples. GitHub repo: [neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts).

The Spaces implementation was created by [mdnestor](https://github.com/mdnestor).

## Usage
1. Select a voice - either by choosing a preset, uploading audio files, or recording via microphone - and click **Confirm voice**. Uploaded and recorded audio is chunked into 10 second segments. Follow the guidelines in the [voice customization guide](https://github.com/neonbjb/tortoise-tts#voice-customization-guide).
2. Choose a model preset (ultra fast/fast/standard/high quality), type the text to speak, and click **Generate**.
""")

    voice_samples = gr.State([])

    with gr.Row():
      with gr.Tab("Choose preset voice"):
        audio1 = gr.Dropdown(voices)
        btn1 = gr.Button("Confirm voice")
        btn1.click(load_preset_voice, [audio1], [voice_samples])

      with gr.Tab("Upload audio"):
        audio2 = gr.File(file_count="multiple")
        btn2 = gr.Button("Confirm voice")
        btn2.click(load_voice_from_files, [audio2], [voice_samples])

      with gr.Tab("Record audio"):
        audio3 = gr.Audio(source="microphone")
        btn3 = gr.Button("Confirm voice")
        btn3.click(load_voice_from_recording, [audio3], [voice_samples])

    def main(voice_samples, text, preset):
      gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=None,
        preset=preset
      )
      torchaudio.save("generated.wav", gen.squeeze(0).cpu(), TORTOISE_SR_OUT)
      return "generated.wav"

    preset = gr.Dropdown(presets, value=preset_default, label="Model preset")
    text = gr.Textbox(label="Text to speak")
    btn = gr.Button("Generate")
    audio_out = gr.Audio()
    btn.click(main, [voice_samples, text, preset], audio_out)

demo.launch()
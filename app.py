import gradio as gr
import os
import sys

os.system("git clone https://github.com/neonbjb/tortoise-tts.git")
sys.path.append("./tortoise-tts/")
os.system("pip install -r ./tortoise-tts/requirements.txt")
os.system("python ./tortoise-tts/setup.py install")

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

tts = TextToSpeech()

def main(text, voice, preset):
  voice_samples, conditioning_latents = load_voice(voice)
  gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, preset=preset)
  torchaudio.save("generated.wav", gen.squeeze(0).cpu(), 24000)
  return "generated.wav"

voices = ["mol", "tom", "applejack", "daniel", "myself", "weaver", "train_empire", "train_dotrice", "rainbow", "pat", "geralt", "halle", "train_kennard", "jlaw", "train_grace", "angie", "william", "tim_reynolds", "train_atkins", "train_dreams", "train_mouse", "freeman", "deniro", "lj", "train_lescault", "emma", "pat2", "snakes", "train_daws"]
presets = ["ultra_fast", "fast", "standard", "high_quality"]

gr.Interface(
  main, 
  [
    gr.Textbox(label="Text", placeholder="Text-to-speak goes here..."),
    gr.Dropdown(voices, value="deniro", label="Voice"),
    gr.Dropdown(presets, value="ultra_fast", label="Preset"),
  ],
  gr.Audio(),
  description="TorToiSe - a multi-voice TTS system | <a href=\"https://github.com/neonbjb/tortoise-tts\">source</a>",
  enable_queue=True
).launch()
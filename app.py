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

TORTOISE_SR     = 22050
TORTOISE_SR_OUT = 24000

tts = TextToSpeech()

import librosa
import soundfile


def split_into_chunks(arr, sr, chunk_duration):
  duration = len(arr) / sr
  num_chunks = 1 + int(duration/chunk_duration)
  chunks = [arr[(sr*chunk_duration*i):(sr*chunk_duration*(i+1))] for i in range(num_chunks)]
  return chunks

voices = ["mol", "tom", "applejack", "daniel", "myself", "weaver", "train_empire", "train_dotrice", "rainbow", "pat", "geralt", "halle", "train_kennard", "jlaw", "train_grace", "angie", "william", "tim_reynolds", "train_atkins", "train_dreams", "train_mouse", "freeman", "deniro", "lj", "train_lescault", "emma", "pat2", "snakes", "train_daws"]
presets = ["ultra_fast", "fast", "standard", "high_quality"]
preset_default = "fast"

with gr.Blocks() as demo:
    gr.Markdown("""# TorToiSe
Tortoise is a text-to-speech model developed by James Betker. It is capable of zero-shot voice cloning from a small set of voice samples. GitHub repo: [neonbjb/tortoise-tts](https://github.com/neonbjb/tortoise-tts).

The Spaces implementation was created by [mdnestor](https://github.com/mdnestor). Currently in alpha; submit issues [here](https://huggingface.co/spaces/mdnestor/tortoise/discussions)!

## Usage
1. Select a voice - either by choosing a preset, uploading audio files, or recording via microphone - and click **Confirm voice**. Uploaded and recorded audio is chunked into 10 second segments. Follow the guidelines in the [voice customization guide](https://github.com/neonbjb/tortoise-tts#voice-customization-guide).
2. Choose a model preset (ultra fast/fast/standard/high quality), type the text to speak, and click **Generate**.
""")

    voice = gr.State([])

    with gr.Row():
      with gr.Tab("Choose preset voice"):
        audio1 = gr.Dropdown(voices)
        btn1 = gr.Button("Select voice")

      with gr.Tab("Upload audio"):
        audio2 = gr.File(file_count="multiple")
        btn2 = gr.Button("Upload")

      with gr.Tab("Record audio"):
        audio3 = gr.Audio(source="microphone")
        btn3 = gr.Button("Upload")

    btn1.click(
        lambda x: x,
        [audio1],
        [voice],
    )

    def btn2_main(inp):
      a = [librosa.load(xi.name) for xi in inp]
      chunks = []
      for x in a:
        chunks += split_into_chunks(x[0], sr=TORTOISE_SR, chunk_duration=10)
      os.system("rm -rf ./tortoise-tts/tortoise/voices/custom/")
      os.system("mkdir ./tortoise-tts/tortoise/voices/custom/")
      for i, chunk in enumerate(chunks):
        soundfile.write(f"./tortoise-tts/tortoise/voices/custom/{i}.wav", chunk, TORTOISE_SR)
      return "custom"

    btn2.click(
        btn2_main,
        [audio2],
        [voice],
    )

    def btn3_main(inp):
      sample_rate, audio = inp
      soundfile.write("recording.wav", audio, sample_rate) # not the most efficient!
      audio, _ = librosa.load("recording.wav")
      chunks = split_into_chunks(audio, sr=TORTOISE_SR, chunk_duration=10)
      os.system("rm -rf ./tortoise-tts/tortoise/voices/custom/")
      os.system("mkdir ./tortoise-tts/tortoise/voices/custom/")
      for i, chunk in enumerate(chunks):
        soundfile.write(f"./tortoise-tts/tortoise/voices/custom/{i}.wav", chunk, TORTOISE_SR)
      return "custom"

    btn3.click(
        btn3_main,
        [audio3],
        [voice],
    )

    def main(voice, text, preset):
      voice_samples, conditioning_latents = load_voice(voice)
      gen = tts.tts_with_preset(
        text,
        voice_samples=voice_samples,
        conditioning_latents=conditioning_latents,
        preset=preset
      )
      torchaudio.save("generated.wav", gen.squeeze(0).cpu(), TORTOISE_SR_OUT)
      return "generated.wav"

    preset = gr.Dropdown(presets, value=preset_default, label="Model preset")
    text = gr.Textbox(label="Text to speak")

    btn = gr.Button("Generate")
    audio_out = gr.Audio()
    
    btn.click(
      main,
      [voice, text, preset],
      audio_out
    )

demo.launch(debug=True)
import os

os.system("git clone https://github.com/jnordberg/tortoise-tts.git .")
os.system("pip install -r requirements.txt")
os.system("python setup.py install")

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

tts = TextToSpeech()
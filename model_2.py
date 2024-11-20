from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import subprocess
import json
import os
import regex as re  # Используем regex для Unicode Property Escapes

# Отключить предупреждение Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

SetLogLevel(0)

if not os.path.exists("model"):
    print("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit(1)

FRAME_RATE = 16000
CHANNELS = 1

model = Model("model")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)

mp3 = AudioSegment.from_mp3('audio.mp3')
mp3 = mp3.set_channels(CHANNELS)
mp3 = mp3.set_frame_rate(FRAME_RATE)

rec.AcceptWaveform(mp3.raw_data)
result = rec.Result()
text = json.loads(result)["text"]

try:
    cased = subprocess.check_output(
        r'python C:\Projects\transcription\recasepunc\recasepunc.py predict C:\Projects\transcription\recasepunc\checkpoint',
        shell=True, text=True, input=text, stderr=subprocess.STDOUT
    )
except subprocess.CalledProcessError as e:
    print(f"Command failed with error code {e.returncode}")
    print(f"Command output: {e.output}")
    raise

with open('data.txt', 'w', encoding='utf-8') as f:
    f.write(cased)

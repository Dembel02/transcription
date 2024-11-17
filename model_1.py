from vosk import Model, KaldiRecognizer, SetLogLevel
from pydub import AudioSegment
import subprocess
import json
import os


# Отключить предупреждение Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

SetLogLevel(0)

# Проверяем наличие модели
if not os.path.exists("model"):
    print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit (1)

# Устанавливаем Frame Rate
FRAME_RATE = 16000
CHANNELS=1

model = Model("model")
rec = KaldiRecognizer(model, FRAME_RATE)
rec.SetWords(True)

# Используя библиотеку pydub делаем предобработку аудио
mp3 = AudioSegment.from_mp3('audio.mp3')
mp3 = mp3.set_channels(CHANNELS)
mp3 = mp3.set_frame_rate(FRAME_RATE)

# Преобразуем вывод в json
rec.AcceptWaveform(mp3.raw_data)
result = rec.Result()
text = json.loads(result)["text"]

# Добавляем пунктуацию
# cased = subprocess.check_output(r'python C:\Projects\transcription\recasepunc\recasepunc.py predict C:\Projects\transcription\recasepunc\checkpoint', shell=True, text=True, input=text)

# try:
#     cased = subprocess.check_output(
#         r'python C:\Projects\transcription\recasepunc\recasepunc.py predict C:\Projects\transcription\recasepunc\checkpoint',
#         shell=True, text=True, input=text
#     )
# except subprocess.CalledProcessError as e:
#     print(f"Command failed with error: {e}")
#     print(f"Command output: {e.output}")
#     raise

# Записываем результат в файл "data.txt"
with open('data.txt', 'w', encoding='utf-8') as f:
    json.dump(text, f, ensure_ascii=False, indent=4)
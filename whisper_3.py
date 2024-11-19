import whisper
from pyannote.audio import Pipeline
import librosa
# huggingface-cli login

# Загрузить модель Whisper
model = whisper.load_model("base")

# Загрузить модель для диаризации (pyannote.audio)
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", 
    use_auth_token="hf18_SbBzzjXRcJDzZOnxJzppqpTIUJYGSAjjoX",
    local_files_only=True)

# Выполнить диаризацию
audio_path = "audioextract.mp3"
diarization = pipeline(audio_path)

# Загрузить аудио
audio, sr = librosa.load(audio_path, sr=16000)

# Словарь для хранения текста по спикерам
speaker_texts = {}

# Обработка каждого сегмента
for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_time = int(turn.start * sr)  # начало в семплах
    end_time = int(turn.end * sr)  # конец в семплах
    segment = audio[start_time:end_time]

    # Убедиться, что длина сегмента соответствует требованиям Whisper
    segment = whisper.pad_or_trim(segment)
    mel = whisper.log_mel_spectrogram(segment).to(model.device)

    # Транскрибировать сегмент
    options = whisper.DecodingOptions(language="Russian")
    result = model.decode(mel, options)

    # Добавить текст в словарь для соответствующего спикера
    if speaker not in speaker_texts:
        speaker_texts[speaker] = ""
    speaker_texts[speaker] += result.text + " "

# Сохранение результатов в файл
output_file = "speaker_transcription.txt"
with open(output_file, "w", encoding="utf-8") as file:
    for speaker, text in speaker_texts.items():
        file.write(f"{speaker}:\n{text.strip()}\n\n")

print(f"Результат сохранен в файл: {output_file}")

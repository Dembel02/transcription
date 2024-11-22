import whisper
import math

# Загрузить модель
model = whisper.load_model("base")

# Загрузить аудио
audio = whisper.load_audio("audioextract.mp3")

# Разделить аудио на сегменты по 30 секунд
segment_duration = 30  # в секундах
sampling_rate = 16000  # частота дискретизации Whisper
segment_length = segment_duration * sampling_rate
total_segments = math.ceil(len(audio) / segment_length)

# Обработка каждого сегмента
full_text = ""

for i in range(total_segments):
    start = i * segment_length
    end = start + segment_length
    segment = audio[start:end]
    segment = whisper.pad_or_trim(segment)  # Убедиться, что сегмент имеет правильную длину

    # Получить спектрограмму и обработать через модель
    mel = whisper.log_mel_spectrogram(segment).to(model.device)
    options = whisper.DecodingOptions(language="Russian")
    result = model.decode(mel, options)

    # Добавить результат к общему тексту
    full_text += result.text + " "

# Вывести полный текст
print(full_text)
# 2024-11-14 13-43-58.mp4
from moviepy.editor import VideoFileClip

video = VideoFileClip(r"2024-11-14 13-43-58.mp4")  # В кавычках указать путь к видео

video.audio.write_audiofile(r"audio.mp3")  # В кавычках указать название итогового аудиофайла
import whisper
from moviepy.editor import VideoFileClip
import os
import subprocess
import numpy as np
import torch
import wave

# Definir o caminho completo para o executável do FFmpeg
ffmpeg_path = r'C:\Users\Makaya.Afonso\Downloads\ffmpeg\bin\ffmpeg.exe'

# Garantir que o FFmpeg está no PATH temporariamente
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# Função para carregar áudio usando FFmpeg
def load_audio(audio_path):
    temp_wav = 'temp_audio.wav'
    cmd = [ffmpeg_path, '-i', audio_path, '-f', 'wav', temp_wav]
    subprocess.run(cmd, capture_output=True, check=True)
    
    # Ler o arquivo WAV temporário e retornar como array NumPy
    with wave.open(temp_wav, 'rb') as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        return audio

# Carregar o vídeo e extrair o áudio
video_path = "replay_de_14_de_marÇo (240p).mp4"
audio_path = "extracted_audio.wav"

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path, codec='pcm_s16le')

# Carregar o modelo Whisper
model = whisper.load_model("base")

# Carregar o áudio e transcrever
audio_data = load_audio(audio_path)
audio_data = torch.from_numpy(audio_data)  # Convertendo para tensor PyTorch

# Transcrever o áudio especificando o idioma
result = model.transcribe(audio_path, language='pt')

# Salvar a transcrição em um arquivo de texto
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcrição concluída e salva em 'transcription.txt'")
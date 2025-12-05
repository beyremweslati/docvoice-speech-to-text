import os
import wave

# Paths
input_dir = "../dataset/processed/audio"
output_dir = "../dataset/processed/audio_final"

os.makedirs(output_dir, exist_ok=True)

channels = 1
sampwidth = 2       # 16-bit
framerate = 16000   # 16 kHz

for filename in os.listdir(input_dir):
    if filename.endswith(".pcm"):
        pcm_path = os.path.join(input_dir, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_dir, wav_filename)

        with open(pcm_path, "rb") as pcmfile:
            pcm_data = pcmfile.read()

        with wave.open(wav_path, "wb") as wavfile:
            wavfile.setnchannels(channels)
            wavfile.setsampwidth(sampwidth)
            wavfile.setframerate(framerate)
            wavfile.writeframes(pcm_data)

        print(f"Converted {filename}: {wav_filename}")

print("All PCM files converted to WAV successfully")

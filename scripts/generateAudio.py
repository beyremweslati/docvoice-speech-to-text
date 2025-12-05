from elevenlabs.client import ElevenLabs 
from elevenlabs import stream, save
from dotenv import load_dotenv
import pandas as pd
import os
import time

load_dotenv()

elevenlabs = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY")
)

input_file = os.path.join("..", "dataset", "medicalTermsStudySet_v1.csv")
output_dir = os.path.join("..", "dataset", "processed", "audio")
transcript_dir = os.path.join("..", "dataset", "processed", "transcripts")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(transcript_dir, exist_ok=True)

df = pd.read_csv(input_file)

terms = df["Example Term"].dropna().drop_duplicates().tolist()

for i, term in enumerate(terms):
    try:
        if i == 0:
            time.sleep(1)
        audio_stream = elevenlabs.text_to_speech.convert(
            text=term,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="pcm_16000",
        )
        audio_path = os.path.join(output_dir, f"{i:03d}.pcm")
        save(audio_stream, audio_path)

        text_path = os.path.join(transcript_dir, f"{i:03d}.txt")
        with open(text_path, "w", encoding="utf-8") as tf:
            tf.write(term)
        print(f"Generated {term}: {audio_path}")
        time.sleep(0.5) 

    except Exception as e:
        print(f"Error generating '{term}': {e}")

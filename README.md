# DocVoice: Ai Speech-To-Text

Developed a medical voice recognition application that assists doctors in writing their reports using voice input. I generated a custom audio dataset from samples of medical texts using ElevenLabs, then fine-tuned the Whisper model to improve recognition of medical terminology. The project includes a Streamlit interface that allows doctors to dictate notes about their patients and instantly receive accurate transcriptions adapted to medical language.

# Workflow Overview

This repository documents the end-to-end process: from generating a specialized medical vocabulary dataset using ElevenLabs and preparing the data, to fine-tuning the Whisper model (using LoRA/PEFT), and finally deploying the model via a live Streamlit interface.

The core process is executed in a sequential four-step pipeline, followed by the final application deployment:

1. generateAudio.py: Creates a list of medical terms (transcripts) and uses ElevenLabs to generate corresponding audio files.

2. WavConverter.py: Converts the generated audio into a consistent .wav format (16kHz, mono) suitable for Whisper training.

3. initialAnalysis.ipynb: Analyzes the dataset, splits it into train.json, val.json, test.json, and runs an initial baseline evaluation using the original Whisper model.

4. fineTuning.ipynb: Applies the LoRA adapter to the base Whisper model using the prepared dataset.

5. app.py: The final Streamlit application for real-time transcription inference.

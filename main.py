import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

RECORD_SECONDS = 5
SAMPLE_RATE = 16000

# Record audio from the microphone
def record_audio():
    print("Recording...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio

# Save audio to a file
def save_audio_to_file(filename, audio):
    write(filename, SAMPLE_RATE, audio)

# Play audio from a file
def play_audio_from_file(filename):
    print("Playing...")
    sample_rate, audio = read(filename)
    sd.play(audio, sample_rate)
    sd.wait()

# Transcribe audio file using Wav2Vec2
def transcribe_audio(filename):
    # Load pre-trained model and tokenizer
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load audio
    sample_rate, audio = read(filename)
    
    # Preprocess audio
    input_values = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    # Perform transcription
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the transcription
    transcription = processor.decode(predicted_ids[0])
    return transcription

def main():
    audio = record_audio()
    save_audio_to_file("recorded_audio.wav", audio)
    play_audio_from_file("recorded_audio.wav")
    transcription = transcribe_audio("recorded_audio.wav")
    print("Transcription:", transcription)

if __name__ == "__main__":
    main()

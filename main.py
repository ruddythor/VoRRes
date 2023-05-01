import pyaudio
import numpy as np
import soundfile as sf
from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor
from scipy.signal import resample_poly

# Initialize ASR model
model = TFWav2Vec2ForCTC.from_pretrained("openai/whisper-tiny")
processor = Wav2Vec2Processor.from_pretrained("openai/whisper-tiny")

# Function to transcribe audio using the ASR model
def transcribe_audio(audio):
    audio = resample_poly(audio, 16000, 44100)
    input_values = processor(audio, return_tensors="tf", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = np.argmax(logits, axis=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Record audio from the microphone
def record_audio(record_seconds=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.hstack(frames)

# Main function
def main():
    while True:
        audio = record_audio()
        transcription = transcribe_audio(audio)
        print("You said:", transcription)

if __name__ == "__main__":
    main()
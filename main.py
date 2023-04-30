import pyaudio
import numpy as np
import soundfile as sf
import requests
import json
import pyttsx3
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

# Function to send text to ChatGPT and get the response
def chat_with_gpt(text):
    api_key = "your_openai_api_key"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps({
        "model": "text-davinci-002",
        "prompt": text,
        "temperature": 0.7,
        "max_tokens": 100,
    })
    response = requests.post("https://api.openai.com/v1/engines/davinci-codex/completions", headers=headers, data=data)
    response_text = response.json()["choices"][0]["text"].strip()
    return response_text

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Record audio from the microphone
def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
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
        response_text = chat_with_gpt(transcription)
        print("ChatGPT says:", response_text)
        text_to_speech(response_text)

if __name__ == "__main__":
    main()


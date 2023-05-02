import speech_recognition as sr
import pyaudio
import numpy as np
import soundfile as sf
import pyttsx3
from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor, GPT2LMHeadModel, GPT2Tokenizer
from scipy.signal import resample_poly

# Initialize ASR model
print("Loading and initializing ASR model ...")
asr_model = TFWav2Vec2ForCTC.from_pretrained("openai/whisper-tiny")
asr_processor = Wav2Vec2Processor.from_pretrained("openai/whisper-tiny")

# Initialize GPT model
print("Loading and initializing GPT model ...")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function to transcribe audio using the ASR model
def transcribe_audio(audio):
    audio = resample_poly(audio, 16000, 44100)
    input_values = asr_processor(audio, return_tensors="tf", padding="longest").input_values
    logits = asr_model(input_values).logits
    predicted_ids = np.argmax(logits, axis=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0]
    return transcription

# Function to send a message to GPT and receive a response
def chat_with_gpt(text):
    input_ids = gpt_tokenizer.encode(f"{text}", return_tensors="tf")
    output = gpt_model.generate(input_ids, max_length=150, num_return_sequences=1)
    response_text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
    response_text = response_text.split("AI:")[-1].strip()
    return response_text

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

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
        response_text = chat_with_gpt(transcription)
        print("AI says:", response_text)
        text_to_speech(response_text)

if __name__ == "__main__":
    main()

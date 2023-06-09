import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor, SpeechT5HifiGan
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, SpeechT5Processor
import tensorflow as tf
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import openai
from transformers import AutoProcessor, SpeechT5ForTextToSpeech
import soundfile as sf
from datasets import load_dataset
from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor, GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf


RECORD_SECONDS = 5
SAMPLE_RATE = 16000

def record_audio():
    print("Recording...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return audio

def save_audio_to_file(filename, audio):
    write(filename, SAMPLE_RATE, audio)

def play_audio_from_file(filename):
    print("Playing...")
    sample_rate, audio = read(filename)
    sd.play(audio, sample_rate)
    sd.wait()

def transcribe_audio(filename):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    sample_rate, audio = read(filename)
    input_values = processor(audio, sampling_rate=sample_rate, return_tensors="tf").input_values
    logits = model(input_values).logits
    predicted_ids = tf.argmax(logits, axis=-1)
    transcription = processor.decode(predicted_ids[0].numpy())
    return transcription

def speak_response(response):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    inputs = processor(text=response, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write("speech.wav", speech.numpy(), samplerate=16000)

# Generate a response using GPT-2
def generate_response(prompt):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2-large")

    # Encode prompt
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    
    # Generate text
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def main():
    audio = record_audio()
    save_audio_to_file("recorded_audio.wav", audio)
    # play_audio_from_file("recorded_audio.wav")
    transcription = transcribe_audio("recorded_audio.wav")
    print("Transcription:", transcription)
    response = generate_response(transcription)
    print("Hank says:", response)
    speak_response(response)

if __name__ == "__main__":
    main()

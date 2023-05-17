import sounddevice as sd
import numpy as np
from transformers import SpeechT5HifiGan
import torch
from transformers import GPT2Tokenizer, SpeechT5ForTextToSpeech, SpeechT5Processor, Wav2Vec2Processor, TFWav2Vec2ForCTC
import openai
from scipy.io.wavfile import write, read
import tensorflow as tf
import soundfile as sf
from datasets import load_dataset
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import argparse

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

def speak_response(response, filename):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    inputs = processor(text=response, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write(filename, speech.numpy(), samplerate=16000)

# Generate a response using GPT-2
def generate_response(prompt):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = TFGPT2LMHeadModel.from_pretrained("distilgpt2")

    # Encode prompt
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    
    # Generate text
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def main():
    #speak_response("I have finished booting. Ready to accept instructions.", "boot.wav")
    #speak_response("Thinking.", "thinking.wav")

    parser = argparse.ArgumentParser()
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.offline:

        print("offline mode enabled")
        data, samplerate = sf.read("boot.wav")
        sd.play(data, samplerate)
        sd.wait()
        audio = record_audio()
        save_audio_to_file("recorded_audio.wav", audio)
    
        thinking, samplerate = sf.read("thinking.wav")
        sd.play(thinking, samplerate)
        sd.wait()
        #play_audio_from_file("recorded_audio.wav")a
        #transcription = "What is the future of space travel?"
        #transcription = "How can a ship fly using antigravity?"
        transcription = transcribe_audio("recorded_audio.wav")
        print("Transcription:", transcription)
        
        sd.play(thinking, samplerate)
        sd.wait()
        response = generate_response(transcription)
        print("Hank says:", response)
        speak_response(response, "speech.wav")

        respond, samplerate = sf.read("speech.wav")
        sd.play(respond, samplerate)
        sd.wait()

        print("offline mode enabled")
    else:
        print("Online mode enabled")

if __name__ == "__main__":
    main()

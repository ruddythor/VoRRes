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



def generate_response(prompt):

    openai.organization = 'fake-org'
    # Load your API key from an environment variable or secret management service
    openai.api_key = 'fake-key'
    openai.Model.list()
    
    response = openai.Completion.create(
        engine="text-davinci-003", #"gpt-3.5-turbo-0301", #engine="gpt-4" # This is assuming that GPT-4 might be named 'text-davinci-004'
        prompt=prompt,
        max_tokens=100
    )

    res = response.choices[0].text.strip()
    return res

    # return response

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

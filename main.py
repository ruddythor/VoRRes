import numpy as np
from transformers import SpeechT5HifiGan
import torch
from transformers import GPT2Tokenizer, SpeechT5ForTextToSpeech, SpeechT5Processor 
import openai

import soundfile as sf
from datasets import load_dataset
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel


RECORD_SECONDS = 5
SAMPLE_RATE = 16000


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
    # audio = record_audio()
    # save_audio_to_file("recorded_audio.wav", audio)
    # play_audio_from_file("recorded_audio.wav")
    #transcription = "What is the future of space travel?"
    transcription = "How can a ship fly using antigravity?"
    print("Transcription:", transcription)
    response = generate_response(transcription)
    print("Hank says:", response)
    speak_response(response)

if __name__ == "__main__":
    main()

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write, read
from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor, GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

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
    model = TFWav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load audio
    sample_rate, audio = read(filename)
    
    # Preprocess audio
    input_values = processor(audio, sampling_rate=sample_rate, return_tensors="tf").input_values
    
    # Perform transcription
    logits = model(input_values).logits
    predicted_ids = tf.argmax(logits, axis=-1)
    
    # Decode the transcription
    transcription = processor.decode(predicted_ids[0].numpy())
    return transcription

# Generate a response using GPT-2
def generate_response(prompt):
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2")

    # Encode prompt
    inputs = tokenizer.encode(prompt, return_tensors="tf")
    
    # Generate text
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def main():
    audio = record_audio()
    save_audio_to_file("recorded_audio.wav", audio)
    play_audio_from_file("recorded_audio.wav")
    transcription = transcribe_audio("recorded_audio.wav")
    print("Transcription:", transcription)
    
    response = generate_response(transcription)
    print("Generated response:", response)

if __name__ == "__main__":
    main()

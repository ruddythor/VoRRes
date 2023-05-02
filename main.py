import numpy as np
import sounddevice as sd
import soundfile as sf
import samplerate as sr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# from datasets import load_dataset
import torch
 
 # load model and tokenizer
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
     
 # load dummy dataset and read soundfiles
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 




# Initialize ASR model
# asr_model = TFWav2Vec2ForCTC.from_pretrained("openai/whisper-tiny")
# asr_processor = Wav2Vec2Processor.from_pretrained("openai/whisper-tiny")

# Function to transcribe audio using the ASR model
def transcribe_audio(audio):
    audio = sr.resample(audio, 1.5, 'sinc_best')
    asr_proc = asr_processor(audio, return_tensors="tf", padding="longest", sampling_rate=16000)
    print(asr_proc)
    input_values = asr_proc.input_values
    logits = asr_model(input_values).logits
    predicted_ids = np.argmax(logits, axis=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0]
    return transcription

# Record audio from the microphone
def record_audio(record_seconds=5):
    RATE = 44100
    CHANNELS = 1
    print("Recording...")
    audio = sd.rec(int(record_seconds * RATE), samplerate=RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    print("Finished recording.")
    return audio.flatten()

# Save audio to a WAV file
def save_audio_to_wav(audio, filename):
    sf.write(filename, audio, 44100)

# Load audio from a WAV file
def load_audio_from_wav(filename):
    audio, samplerate = sf.read(filename, dtype='float32')
    return audio.flatten()

def play_audio(audio):
    RATE = 44100
    print("Playing audio...")
    sd.play(audio, samplerate=RATE)
    sd.wait()
    print("Finished playing audio.")

# Main function
def main():
    audio = record_audio()
    save_audio_to_wav(audio, 'audio_input.wav')
    audio_from_wav = load_audio_from_wav('audio_input.wav')
    play_audio(audio_from_wav)    
    # transcription = transcribe_audio(audio_from_wav)
     # tokenize
    input_values = processor(audio_from_wav, return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1
    
    # retrieve logits
    logits = model(input_values).logits
    
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    print("Transcription:", transcription)

if __name__ == "__main__":
    main()

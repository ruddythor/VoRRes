from transformers import TFWav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import tensorflow as tf
import numpy as np


import speech_recognition as sr

def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        speech = recognizer.recognize_google(audio)
        return speech

    except sr.RequestError:
        print("API unavailable")
    except sr.UnknownValueError:
        print("Unable to recognize speech")



def init():

    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    print(microphone)


    speech = recognize_speech_from_mic(recognizer, microphone)

    print(f"You said: {speech}")
    # Load the model and processor
    model = TFWav2Vec2ForCTC.from_pretrained("openai/whisper-tiny")
    processor = Wav2Vec2Processor.from_pretrained("openai/whisper-tiny")

    # # Read the audio file
    # audio, sample_rate = sf.read("path/to/your/audio/file.wav")

    # # Resample the audio if the sample rate is not 16 kHz
    # if sample_rate != 16000:
    #     from scipy.signal import resample_poly
    #     audio = resample_poly(audio, 16000, sample_rate)

    # # Tokenize the audio
    input_values = processor(speech, return_tensors="tf", padding="longest").input_values

    # # Perform speech recognition
    logits = model(input_values).logits

    # # Decode the logits to text
    predicted_ids = tf.argmax(logits, axis=-1).numpy()
    transcription = processor.batch_decode(predicted_ids)[0]

    # # Print the transcription
    print(transcription)

if __name__ == "__main__":
    init()
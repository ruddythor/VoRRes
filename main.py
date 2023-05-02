import speech_recognition as sr
import requests
import pyttsx3
import json

# Constants
OPENAI_API_KEY = "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"
OPENAI_API_URL = "https://api.openai.com/v1/completions"

# Function to transcribe audio using the SpeechRecognition package
def transcribe_audio(recognizer, audio):
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Function to send a message to ChatGPT and receive a response
def chat_with_gpt(text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": f"{text}",
        "model": "text-davinci-003",
        "max_tokens": 150,
        "temperature": 0.5,
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=data)
    print(response)
    response_text = json.loads(response.text)["choices"][0]["text"].strip()
    return response_text

# Function to convert text to speech
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Main function
def main():
    recognizer = sr.Recognizer()
    while True:
        with sr.Microphone() as source:
            print("Recording...")
            audio = recognizer.listen(source)
            print("Finished recording.")
            transcription = transcribe_audio(recognizer, audio)
            if transcription:
                print("You said:", transcription)
                response_text = chat_with_gpt(transcription)
                print("ChatGPT says:", response_text)
                text_to_speech(response_text)

if __name__ == "__main__":
    main()

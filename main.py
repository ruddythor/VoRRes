import speech_recognition as sr

# Function to transcribe audio using the SpeechRecognition package
def transcribe_audio(recognizer, audio):
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

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

if __name__ == "__main__":
    main()
# VoRRes
This is the Voice Receive/Responder Module for AI projects. It takes voice/audio input, translates that input to commands to delegate to other AIs, then returns an audio response to the user stating its actions taken.

This project should be modular, and installable in any other Robot project to provide voice capabilities. Hooking it up to other hardware as a controller is not currently supported. It is more of a "voice assistant installation" for robots.

# Installation
* need a simple single-line command to run the daemon 

# Development
Workflow for this project is to develop code on your windows machine, push to github, pull from github on the raspberry pi, then run code. cannot currently get the environments right across windows and linux to get a simplified development experience across platforms atm, so this should make the process simpler for now and we can get more platform stuff working later.


For Raspberry Pi:
* install VS Code 2
* set up Venv, install requirements: 
    * on windows: `mkdir venv && python -m venv venv`
    * `pip install -r requirements.txt`
    * may have to `pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl` for the right tensorflow install.

# Convert to Tensorflow Lite
I think this is something that will only have to be done at the repo level, and we should commit the resulting .tflite file/model to the repo.

```from transformers import TFWav2Vec2ForCTC

model = TFWav2Vec2ForCTC.from_pretrained("openai/whisper-tiny")
model.save_pretrained("whisper_saved_model", saved_model=True)```

```import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("whisper_saved_model")
tflite_model = converter.convert()

with open("whisper_tflite_model.tflite", "wb") as f:
    f.write(tflite_model)```


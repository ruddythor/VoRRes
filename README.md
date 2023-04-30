# VoRRes
This is the Voice Receive/Responder Module for AI projects. It takes voice/audio input, translates that input to commands to delegate to other AIs, then returns an audio response to the user stating its actions taken.

This project should be modular, and installable in any other Robot project to provide voice capabilities. Hooking it up to other hardware as a controller is not currently supported. It is more of a "voice assistant installation" for robots.

# Installation
* need a simple single-line command to run the daemon 

# Development
For Raspberry Pi:
* install VS Code 2
* set up Venv, install requirements: 
    * on windows: `mkdir venv && python -m venv venv`
    * `pip install -r requirements.txt`

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


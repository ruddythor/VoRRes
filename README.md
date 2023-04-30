# VoRRes
This is the Voice Receive/Responder Module for AI projects. It takes voice/audio input, translates that input to commands to delegate to other AIs, then returns an audio response to the user stating its actions taken.

# Installation
* need a simple single-line command to run the daemon 

# Development
For Raspberry Pi:
* install VS Code 2
* set up Venv, install requirements: 
    * pip install tensorflow
    * pip install transformers
    * pip install soundfile scipy

# Convert to Tensorflow Lite
```from transformers import TFWav2Vec2ForCTC

model = TFWav2Vec2ForCTC.from_pretrained("openai/whisper-tiny")
model.save_pretrained("whisper_saved_model", saved_model=True)```

```import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("whisper_saved_model")
tflite_model = converter.convert()

with open("whisper_tflite_model.tflite", "wb") as f:
    f.write(tflite_model)```


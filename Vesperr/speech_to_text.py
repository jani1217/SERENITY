import transformers
print(transformers.__version__)

import librosa
import torch
import IPython.display as display
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import numpy as np

from google.colab import drive
drive.mount('/content/drive')


#load pre-trained model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


#load audio file
audio, sampling_rate = librosa.load("/content/test2.opus",sr=16000)

audio,sampling_rate


# audio
display.Audio("/content/test2.opus", autoplay=True)

input_values = tokenizer(audio, return_tensors = 'pt').input_values
input_values

logits = model(input_values).logits
logits

predicted_ids = torch.argmax(logits, dim =-1)

transcriptions = tokenizer.decode(predicted_ids[0])

transcriptions
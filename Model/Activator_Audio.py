import MLMainCode
import pandas as pd 
import os
from pathlib import Path
import glob
import numpy as np
from Model.MLMainCode import mainFunc
import audioToText

path = Path.cwd().parent
extension = ('mp3', 'wav', 'aac', 'flac') #Can be changed to include JSON
path = path / 'aud_data'
os.chdir(path)
audioList = glob.glob('*.{}'.format(extension))
if not audioList:
    raise Exception("DATA NOT FOUND")

df = pd.DataFrame(columns=['id', 'text', 'label'])
id = []
text = []
for audio_id in audioList:
    id.append(audio_id)
    text.append(audioToText(path / audio_id))

df['id'] = id
df['text'] = text

df.to_csv('Model\input\Input.csv')

mainFunc()
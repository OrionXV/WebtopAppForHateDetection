import MLMainCode
import pandas as pd 
import os
from pathlib import Path
import glob
import numpy as np
#from Model.MLMainCode import mainFunc

"""from Model.imageToText""" 
import imageToText

path = Path.cwd()
extension = ('jpg', 'jpeg', 'png', 'gif') #Can be changed to include JSON
path = path / 'img_data'
os.chdir(path)
imagesList = glob.glob('*.{}'.format(extension))
if not imagesList:
    raise Exception("DATA NOT FOUND")

df = pd.DataFrame(columns=['id', 'text', 'label'])
id = []
text = []
for image_id in imagesList:
    id.append(image_id)
    text.append(imageToText(path / image_id))

df['id'] = id
df['text'] = text

df.to_csv('Model\input\Input.csv')

MLMainCode.mainFunc()
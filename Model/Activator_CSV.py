import MLMainCode
import pandas as pd 
import os
from pathlib import Path
import glob
import numpy as np

from Model.MLMainCode import mainFunc

path = Path.cwd().parent
extension = 'csv' #Can be changed to include JSON
path = path / 'input'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
if not result:
    raise Exception("DATA NOT FOUND")

data_path = path + '/' + result[0]
data = pd.read_csv(data_path)

#data['label'] = np.nan

data.to_csv(path / 'Input.csv', index = True)

mainFunc()
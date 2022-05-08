"""from Model.MLMainCode""" 
"""from Model.generalCase """
import generalCase
import MLMainCode
from pathlib import Path

path = Path.cwd()
print(path)

f = open('Model\\temp\\url.txt', 'r')
link = f.readline()

generalCase.generalCaseScraping(link)
MLMainCode.mainFunc()
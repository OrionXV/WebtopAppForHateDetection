from Model.MLMainCode import mainFunc
from Model.generalCase import generalCaseScraping
import generalCase
import MLMainCode

f = open("Model\\temp\\url.txt", 'r')
link = f.readline()

generalCaseScraping(link)
mainFunc()
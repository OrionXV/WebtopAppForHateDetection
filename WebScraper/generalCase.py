#!pip instal PyPDF2
#!pip install pdfkit
#!sudo apt-get install wkhtmltopdf

import re 
import pandas as pd
from urllib.error import HTTPError, URLError
import pdfkit
import pathlib as pl
import os
import PyPDF2

def get_files(folder):
    import os
    os.chdir(folder)
    files = os.listdir()
    files = [x for x in files if x.endswith(".pdf")]
    return files 


def scrapper(text):
    #, tag = '<p>'
    #pattern = re.compile("{}.*".format(tag))
    #m = pattern.findall(text) 
    #m = re.sub(r'<.*>', '', text)
    df = pd.DataFrame(columns=['id', 'text'])
    text_data = []
    id_data = []
    for i, x in enumerate(text):
        id_data.append(i)
        text_data.append(x)
    df['id'] = id_data
    df['text'] = text_data

    return df

website = input("Enter the website with http/https tag :")
config = pdfkit.configuration(wkhtmltopdf='C:\Program Files\wkhtmltopdf\\bin\wkhtmltopdf.exe')
try:
    pdfkit.from_url(website, "WebScraper/website.pdf", configuration=config) 
except HTTPError as e:
    print(e)
except URLError:
    print("Server down or Incorrect domain")

else:
    pdffileobj=open("WebScraper/website.pdf",'rb')
    
    pdfreader=PyPDF2.PdfFileReader(pdffileobj)
    
    comp = []
    for page in pdfreader.pages:
        print(page.extractText())
        text=page.extractText()
        comp.append(text)

    final_frame = scrapper(comp)
    final_frame.to_csv('Model\input\sampleInput.csv', index= True)
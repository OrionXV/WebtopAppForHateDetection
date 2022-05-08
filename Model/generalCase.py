#!pip instal PyPDF2
#!pip install pdfkit
#!sudo apt-get install wkhtmltopdf

import re 
import pandas as pd
from urllib.error import HTTPError, URLError
import pdfkit
import pathlib as pl
import os
#import PyPDF2
import pdfplumber

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
    for i, x in enumerate(text.split('\n')):
        id_data.append(i)
        text_data.append(x)
    df['id'] = id_data
    df['text'] = text_data

    return df
def generalCaseScraping(link):
    website = link #input("Enter the website with http/https tag :")
    #config = pdfkit.configuration(wkhtmltopdf='C:\Program Files\wkhtmltopdf\\bin\wkhtmltopdf.exe')
    try:
        pdfkit.from_url(website, "WebScraper/temp/website.pdf") #, configuration=config) 
    except HTTPError as e:
        print(e)
    except URLError:
        print("Server down or Incorrect domain")

    else:
        
        
        with pdfplumber.open(r"WebScraper/temp/website.pdf") as pdf:
            first_page = pdf.pages[0]
            print(first_page.extract_text())
            #comp = []
            #for page in pdf.pages:
                #print(page.extractText())
                #page = pdf.pages[i]
                #text = page.extractText()
            #comp.append()
            text = first_page.extract_text()
        
        final_frame = scrapper(text)
        final_frame.to_csv('Model\input\sampleInput.csv', index= True)

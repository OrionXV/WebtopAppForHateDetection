import re
import urllib.request   
import pandas as pd
from urllib.error import HTTPError, URLError

def scrapper(text):
    #tag = '<p>'
    #pattern = re.compile("{}.*".format(tag))
    #m = pattern.findall(text) 
    m = re.sub(r'<.*>', '', text)
    df = pd.DataFrame(columns=['id', 'text'])
    text_data = []
    id_data = []
    for i, x in enumerate(m):
        id_data.append(i)
        text_data.append(x)
    df['id'] = id_data
    df['text'] = text_data

    return df

website = input("Enter the website with http/https tag :")

try:
    urllib.request.urlretrieve(website, "WebScraper/website.txt")
except HTTPError as e:
    print(e)
except URLError:
    print("Server down or Incorrect domain")

else:
    f = open("WebScraper/website.txt", "r", encoding="utf8")
    comp = ''
    for x in f:
        comp = comp + x
    f.close()
    final_frame = scrapper(comp)
    final_frame.to_csv('Model\input\sampleInput.csv', index= True)
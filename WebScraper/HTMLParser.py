import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

def install(package):
    subprocess.check_call([sys.executable,"-m","pip","install",package])


from bs4 import BeautifulSoup
import requests
import html5lib


url = input("Enter te website to scrap : ")
url = "https://"+url

r = requests.get(url)
html = r.content

s = BeautifulSoup(html, 'html.parser')

print(s.get_text())

import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def install(package):
    subprocess.check_call([sys.executable,"-m","pip","install",package])

install("beautifulsoup4")
from bs4 import BeautifulSoup  #Scraper for HTML elements

install("selenium")
from selenium import webdriver #Scraper for JS elements

try:
    website = input("Enter the website with http/https tag :") #Website Input
    html = urlopen(website)
except HTTPError as e:
    print(e)
except URLError:
    print("Server down or Incorrect domain")
else:
    res = BeautifulSoup(html.read(),"html5lib")
    tags = res.findAll("h2",{"class":"widget-title"})
    for tag in tags:
        print(tag.getText())

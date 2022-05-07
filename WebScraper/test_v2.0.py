import http.client as hc
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def install(package):
    subprocess.check_call([sys.executable,"-m","pip","install",package])

try:
    website = input("Enter the website to scrap : ")
    api_key = "qFhxlROo20nLGNfLU0SYgDi11JCo937e"

    conn = hc.HTTPSConnection("api.webscrapingapi.com")

    conn.request("GET", "/v1?api_key="+api_key+"&url=https%3A%2F%2F"+website+"%2Fget")

    res = conn.getresponse()
    data = res.read()

    print(data.decode("utf-8"))
except HTTPError as e:
    print(e['error'])
except URLError:
    print("Server not found")
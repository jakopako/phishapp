import base64
import requests

with open("C:/Users/jakob/data/phishing/test/postfinance.png", "rb") as img_file:
    img_string = base64.encodebytes(img_file.read()).decode()

ip = "192.168.99.100"
# ip = "127.0.0.1"
r = requests.post('http://{}:5000/predict'.format(ip), json={"image": img_string})

print(r.status_code)
print(r.text)
import requests

URL = "http://127.0.0.1:5000/analyze_tweet/"
DATA ={
    'keyword': ['aaa'], 
    'location': ['NewsYork'], 
    'text': ["Our Deeds are the Reason of this #earthquake M"]
}

resp = requests.post(URL, json={**DATA})
print(resp.text)

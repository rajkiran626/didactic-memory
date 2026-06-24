import requests

url = "http://127.0.0.1:5000/recognize"

with open("test.jpg", "rb") as f:

    files = {
        "image": f
    }

    response = requests.post(
        url,
        files=files
    )

print(response.text)
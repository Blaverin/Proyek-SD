import requests

url = 'https://raw.githubusercontent.com/Blaverin/File-proyek-PSD/main/crime.csv'
response = requests.get(url)

with open('crime.csv', 'wb') as file:
    file.write(response.content)

print("Download complete!")

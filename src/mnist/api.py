import requests

pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)

if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')
import requests

url = "https://prod-00.uksouth.logic.azure.com:443/workflows/e5982dd4f7744501b4487fcced2a17c4/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=kDCIlWa4-bquIgnVnSo3Nl3f2mRmQHjjCfd_QKiYznQ"
payload="{\r\n\"orders\":\r\n[\r\n384329\r\n]\r\n}\r\n"
headers = {
  'Content-Type': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)
print(response.status_code)
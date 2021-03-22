import requests as re
import json
import pandas as pd

#Json = re.get("https://mtgjson.com/api/v5/Standard.json")


a = open('./data/StandardAtomic.json', "r", encoding='utf-8')
js = json.load(a)['data']

CardNames = list(js.keys())
print(CardNames)
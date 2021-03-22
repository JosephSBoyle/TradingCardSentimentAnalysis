from readability import Document
from colorama import Style, Fore

import requests
import re
import json

from bs4 import BeautifulSoup
from bs4.element import NavigableString
import html2text

your_text = """I was surfing <a href="...">www.google.com</a>, and I found an
interesting site https://www.stackoverflow.com/. It's amazing! I also liked
Heroku (http://heroku.com/pricing)
more.domains.tld/at-the-end-of-line
https://at-the_end_of-text.com"""


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, "", raw_html)
    cleantext = " ".join(cleantext.splitlines())
    return cleantext


a = open('./data/StandardAtomic.json', "r", encoding='utf-8')
js = json.load(a)['data']

CardNames = list(js.keys())
print(CardNames)

one = "https://www.mtggoldfish.com/articles/against-the-odds-tergrid-god-of-fright-standard"
two = "https://strategy.channelfireball.com/all-strategy/mtg/channelmagic-articles/theros-beyond-death-constructed-set-review-green/"
three = "https://article.hareruyamtg.com/article/48018/?lang=en"

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
response = requests.get(two, headers=headers)
doc = Document(response.text)

h = html2text.HTML2Text()
h.ignore_links = True
text = h.handle(response.text)

print(text)

cleaned_content = cleanhtml(doc.summary())
sentences = text.split(".")


for sent in sentences:
    for card in CardNames:
        match = sent.find(card)
        if match != -1:
            le = len(card)
            x = match.__index__()
            print(f"\n{10*'#'}")
            print("Match:", f"{Fore.RED}{sent[x:x + le]}{Style.RESET_ALL}")
            print("Corpus:", sent[:x-1], f"{Fore.RED}{sent[x:x + le]}{Style.RESET_ALL} {sent[x+le+1:]}.", "\n")
            print(f"{10*'#'}\n")

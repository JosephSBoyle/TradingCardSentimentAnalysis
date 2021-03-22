import requests
from bs4 import BeautifulSoup


def get_review_urls():
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    pages = ["https://strategy.channelfireball.com/all-strategy/tag/constructed-set-review/"] + [f"https://strategy.channelfireball.com/all-strategy/tag/constructed-set-review/page/{x}/" for x in range(2, 7)]
    limited_pages = ["https://strategy.channelfireball.com/all-strategy/tag/limited-set-review/"] + [f"https://strategy.channelfireball.com/all-strategy/tag/limited-set-review/page/{x}/" for x in range(2, 9)]
    set_review_urls = []

    for x in pages + limited_pages:
        r = requests.get(x, headers=headers)
        for link in BeautifulSoup(r.text, 'lxml').findAll('a'):
            try:
                if ("constructed-set-review" in link.get('href')) or ("limited-set-review" in link.get('href')):
                    print(link)
                    set_review_urls.append(str(link.get('href')))
            except TypeError:
                pass
    return set_review_urls


if __name__ == '__main__':
    x = get_review_urls()
    print(f"There are {len(x)} total set reviews, of which {len(set(x))} are unique")

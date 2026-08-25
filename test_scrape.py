from learning.web_learner import WebScraper
import re
from bs4 import BeautifulSoup

scraper = WebScraper()
html = scraper.session.get("https://id.wikipedia.org/wiki/Kecerdasan_buatan").text
extractor = scraper.extractor
soup = BeautifulSoup(html, 'html.parser')

for tag_name in extractor.REMOVE_TAGS:
    for tag in soup.find_all(tag_name):
        tag.decompose()

for cls in extractor.REMOVE_CLASSES:
    for tag in soup.find_all(class_=re.compile(cls, re.I)):
        if tag.name not in ['html', 'body', 'main', 'article']:
            tag.decompose()

main_content = soup.find('article') or soup.find('main')
content_text = ""
if main_content:
    content_text = extractor._extract_text(main_content)

content_text = extractor._clean_text(content_text)
print("Extracted word_count:", len(content_text.split()))


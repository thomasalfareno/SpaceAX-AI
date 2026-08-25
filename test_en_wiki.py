from learning.web_learner import WebScraper
from bs4 import BeautifulSoup

scraper = WebScraper()
html = scraper.session.get("https://en.wikipedia.org/wiki/Tangent_half-angle_substitution").text
extractor = scraper.extractor
soup = BeautifulSoup(html, 'html.parser')

main_content = soup.find('article') or soup.find('main')
print("main_content exists:", main_content is not None)
print("main_content children types:", [type(c).__name__ for c in main_content.children])
print("main_content children tag names:", [c.name for c in main_content.children if hasattr(c, 'name')])

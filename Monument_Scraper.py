# Source of data: https://en.wikipedia.org/wiki/Monuments_of_National_Importance_(India)

import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_Monuments_of_National_Importance_in_Bihar"

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'class': 'wikitable'})

monument_names = []

for row in table.find_all('tr')[1:]:  
    cells = row.find_all('td')
    if len(cells) > 1:
        monument_name = cells[0].get_text(strip=True)
        monument_names.append(monument_name)

with open('monuments.txt', 'a') as file:
    for i, name in enumerate(monument_names, 1):
        file.write(f"{i}. {name}\n")

print("Monument names have been saved to 'monuments.txt'.")

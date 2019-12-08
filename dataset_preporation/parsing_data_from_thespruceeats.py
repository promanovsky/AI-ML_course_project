import requests
from bs4 import BeautifulSoup as soup
from selenium import webdriver
import os
import json
from common.tools import replacing_dict, measures, uncodeDecode, getNumber, findMeasure

def parseRecipeFromUrl(url):
    result = requests.get(url)
    page = result.text
    for k in replacing_dict:
        page = page.replace(k, replacing_dict[k])

    doc = soup(page, 'html.parser')
    recipe_name = uncodeDecode(doc.find_all('h1', {'class':'heading__title'})[0].text)
    rating = float(len(doc.select("div.aggregate-star-rating__stars > a.active"))) + float(len(doc.select("div.aggregate-star-rating__stars > a.half")))*0.5

    ingredients = doc.select("ul#simple-list_1-0 > li.ingredient.simple-list__item")
    found_ingredients_with_amounts = dict()
    for ingr in ingredients:
        row = uncodeDecode(ingr.text.strip().lower())
        words = row.split(' ')
        amount = getNumber(words[0])
        if amount>0:
            am2 = getNumber(words[1])
            if am2>0:
                amount = amount+am2
                row = row.replace(words[1], '')
            measure = findMeasure(row, measures)
            row = uncodeDecode(row.replace(words[0], '').replace(measure, '').strip())
            #print(row)
            found_ingredients_with_amounts[row] = float(amount) * measures[measure]

    #print(recipe_name, rating, found_ingredients_with_amounts)
    return recipe_name, rating, found_ingredients_with_amounts

par_dir = os.path.abspath(os.pardir)


recipe_urls = []
url_base = 'https://www.thespruceeats.com/search?q=&page={}&searchType=recipe&fh=eyJmYWNldHMiOnsicG9wX3NlYXJjaCI6W10sImdyb3VwX2NvdXJzZSI6W10sImdyb3VwX2N1aXNpbmUiOltdLCJncm91cF9kaXNoIjpbeyJ2YWx1ZSI6ImNvY2t0YWlsIiwiZGlzcGxheU5hbWUiOiJjb2NrdGFpbCJ9XSwiZ3JvdXBfcHJlcGFyYXRpb24iOltdLCJncm91cF9vY2Nhc2lvbiI6W10sInN0YXJSYXRpbmdfc2NvcmUiOltdfX0%3D'

for pageNum in range(1, 44):
    url = url_base.format(pageNum)
    with webdriver.PhantomJS() as browser:
        browser.get(url)
        html = browser.page_source
        doc = soup(html, 'html.parser')
        cards = doc.select('a.card')
        for card in cards:
            recipe_urls.append(card['href'])

print('Combined urls recipe count',len(recipe_urls))

rating_data = dict()
ingredients_data = dict()

for url in recipe_urls:
    recipe_name, rating, found_ingredients_with_amounts = parseRecipeFromUrl(url)
    rating_data[recipe_name] = rating
    ingredients_data[recipe_name] = found_ingredients_with_amounts


with open(par_dir + '/datasets/thespruceeats/ingredients_data.json', 'w') as fp:
    json.dump(ingredients_data, fp)

with open(par_dir + '/datasets/thespruceeats/rating_data.json', 'w') as fp:
    json.dump(rating_data, fp)

print('done')
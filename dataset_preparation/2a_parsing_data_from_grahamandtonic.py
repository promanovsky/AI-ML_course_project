import requests
from bs4 import BeautifulSoup as soup
from selenium import webdriver
import os
import json

from common.tools import replacing_dict, measures, uncodeDecode, getNumber, findMeasure

"""
Этап 2a
Граббинг данных с сайта https://www.grahamandtonic.com, парсинг данных в единый формат, сохранение данных.
На основе верстки сайта скрипт проходит по каждой странице пагинатора и сначала собирает ссылки на индивидуальные рецепты.
После сборки ссылок скрипт прогружает каждый рецепт и опираясь на верстку собирает данные по рецепту и рейтингу коктейля.
Собранные данные сохраняются в формате JSON.
"""

par_dir = os.path.abspath(os.pardir)
recipe_urls = []

browser = webdriver.PhantomJS()
browser.get('https://www.grahamandtonic.com/app/htmlsitemap')
html = browser.page_source
doc = soup(html, 'lxml')
browser.close()

links = doc.select("div.col-md-2 > p > a")

link_part = 'https://www.grahamandtonic.com/recipes/all'
for link in links:
    if str(link['href']).startswith(link_part) :
        recipe_urls.append(link['href'])

print('Combined urls recipe count',len(recipe_urls))

def parseRecipeFromUrl(url):
    result = requests.get(url)
    recipe_name, rating, found_ingredients_with_amounts = None, None, dict()

    if result.status_code != 500:
        page = result.text
        for k in replacing_dict:
            page = page.replace(k, replacing_dict[k])

        doc = soup(page, 'html.parser')
        #recipe_name = uncodeDecode(doc.select('h4#RecipeName')[0].text)
        recipe_name = uncodeDecode(doc.select('h4.recipe__name')[0].text)

        rating = float(doc.select('div.col-lg-3.col-md-4.col-sm-12')[0].findChildren("p", recursive=False)[0].text.replace('Average Rating: ', ''))

        #ingredients = doc.select("table#tbl__ingredients > tbody > tr")
        ingredients = doc.select("table.tbl__ingredients > tbody > tr")
        for row in ingredients:
            ingr_name = uncodeDecode(row.findChildren("td", recursive=False)[0].text.strip().lower())
            ingr_amount = uncodeDecode(row.findChildren("td", recursive=False)[1].text.strip().lower())
            ingr_part = uncodeDecode(row.findChildren("td", recursive=False)[2].text.strip().lower())
            if len(ingr_part)>0:
                ingr_name = ingr_name +' ' + ingr_part
            words = ingr_amount.split(' ')
            #print(ingr_name, ingr_amount)
            if len(words)>1:
                amount = getNumber(words[0])
                if amount>0:
                    am2 = getNumber(words[1])
                    if am2>0:
                        amount = amount+am2
                        ingr_amount = ingr_amount.replace(words[1], '')
                    measure = findMeasure(ingr_amount, measures)
                    found_ingredients_with_amounts[ingr_name] = float(amount) * measures[measure]
            else:
                amount = getNumber(words[0])
                if amount>0:
                    found_ingredients_with_amounts[ingr_name] = float(amount) * measures['ounce']

    #print(recipe_name, rating, found_ingredients_with_amounts)
    return recipe_name, rating, found_ingredients_with_amounts

rating_data = dict()
ingredients_data = dict()

for url in recipe_urls:
    recipe_name, rating, found_ingredients_with_amounts = parseRecipeFromUrl(url)
    if not recipe_name is None:
        rating_data[recipe_name] = rating
        ingredients_data[recipe_name] = found_ingredients_with_amounts


with open(par_dir + '/datasets/grahamandtonic/ingredients_data.json', 'w') as fp:
    json.dump(ingredients_data, fp)

with open(par_dir + '/datasets/grahamandtonic/rating_data.json', 'w') as fp:
    json.dump(rating_data, fp)

print('done')


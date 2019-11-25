import pandas as pd
import matplotlib.pyplot as plt
from random import uniform
import re
import os
import json

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + '/datasets/dataset_step_3.csv')
print(df.shape)
print(df.head())

with open(curr_dir + '/datasets/thespruceeats/ingredients_data.json', 'r') as fp:
    ingredients_data_1 = json.load(fp)

with open(curr_dir + '/datasets/grahamandtonic/ingredients_data.json', 'r') as fp:
    ingredients_data_2 = json.load(fp)

with open(curr_dir + '/datasets/thespruceeats/rating_data.json', 'r') as fp:
    rating_data_1 = json.load(fp)

with open(curr_dir + '/datasets/grahamandtonic/rating_data.json', 'r') as fp:
    rating_data_2 = json.load(fp)

used_ingredients = set(df.columns[2::])

new_ingredients = set()
total_ingredients = set() # для наглядности - для работы не нужно - можно удалить

def combineNewIngredients(data, new, total, used):
    for recipe in data.keys():
        for ing in data[recipe].keys():
            if not ing in used:
                new.add(ing)
            total.add(ing)

combineNewIngredients(ingredients_data_1, new_ingredients, total_ingredients, used_ingredients)
combineNewIngredients(ingredients_data_2, new_ingredients, total_ingredients, used_ingredients)

print(len(used_ingredients), used_ingredients)
print(len(total_ingredients), total_ingredients)
print(len(new_ingredients), new_ingredients)

for ingr in new_ingredients:
    zeros = [0 for x in range(df.shape[0])]
    df[ingr] = zeros

rates = [uniform(2,4) for x in range(df.shape[0])]
df = df.assign(rating=rates)

print(df.shape)
print(df.head())

def appendNewRows(data, df, rating_data) -> df:
    zeros = [0 for x in range(df.shape[1] - 1)]
    for recipe_name in data.keys():
        row_list = [recipe_name]
        row_list.extend(zeros)
        row_list[df.columns.get_loc('rating')] = rating_data[recipe_name]
        for ingr in data[recipe_name].keys():
            row_list[df.columns.get_loc(ingr)] = data[recipe_name][ingr]
        #print(row_list)
        row_df = pd.DataFrame([row_list], columns=list(df.columns))
        df = df.append(row_df, ignore_index=True)
    return df

df = appendNewRows(ingredients_data_1, df, rating_data_1)
df = appendNewRows(ingredients_data_2, df, rating_data_2)
print(df.shape)

df = df.loc[:, (df != 0).any(axis=0)]
print(df.shape)

df.to_csv(curr_dir +'/datasets/final_dataset.csv')
print('done')
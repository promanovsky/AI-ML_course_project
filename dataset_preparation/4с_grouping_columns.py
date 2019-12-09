import pandas as pd
import os
from common.tools import ingredients_transformation, find_group_for_ingredient

"""
Этап 4с
Полученные колонки сгруппированы по смыслу в 37 категорий, их значения нужно смержить
"""

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)

par_dir = os.path.abspath(os.pardir)
df = pd.read_csv(par_dir + '/datasets/reducing_dataset.csv')
print(df.shape)

columns_to_group = list(df.columns)
columns_to_group.remove('Cocktail Name')
columns_to_group.remove('rating')

def check_column(column, df):
    if not column in columns_to_group:
        zeros = [0 for x in range(df.shape[0])]
        df[column] = zeros

del_ingredients = set()
for key in ingredients_transformation.keys():
    check_column(key, df)
    for val in ingredients_transformation[key]:
        if val in columns_to_group:
            df[key] = df[key] + df[val]
            del_ingredients.add(val)

print(len(del_ingredients), len(columns_to_group))

df = df.drop(del_ingredients, axis=1)
grupped_columns = list(df.columns)
print(grupped_columns)
grupped_columns.remove('rating')
grupped_columns.append('rating')
print(grupped_columns)
df = df[grupped_columns]

df.to_csv(par_dir +'/datasets/grouped_columns.csv', index=False)
print(df.shape)
print('done')
# (4326, 38)
#print(find_group_for_ingredient('stolichnaya vodka'),find_group_for_ingredient('whiskey barrel bitters'))
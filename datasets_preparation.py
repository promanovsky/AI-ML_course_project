import pandas as pd
import re
import os

from common.tools import replacing_dict, measures, getNumber, formatMeasureValues

curr_dir = os.path.abspath(os.curdir)
# DATASET №1 ====================================================
print('>>> PROCESSING DATASET #1')
df = pd.read_csv(curr_dir + '/datasets/hotaling_cocktails - Cocktails.csv')
print(df.shape)
print(df.head())

df = df.drop(['Bartender','Bar/Company', 'Location', 'Glassware', 'Notes', 'Garnish', 'Preparation'], axis=1)
print(df.shape)
print(df.head())

df['Ingredients'] = df['Ingredients'].str.replace('*','')
df['Ingredients'] = df['Ingredients'].str.lower()

#print(df[df['Ingredients'].str.contains('Vodka'.lower())]) # пример поиска по содержимому в ячейке

ingredients = set()
measures_set = set()

def parseIngredients(source, ingr, measures_set):
    #print('Whole line==> ', source)
    words = re.split('\s', source)
    if len(words)>2:
        measures_set.add(words[1])
        ingr.add(' '.join(words[2::]))

for text in df['Ingredients']:
    for val in text.split(', '):
        parseIngredients(val, ingredients, measures_set)
        #print('===========\n')

print(len(measures_set), measures_set)
print(len(ingredients), ingredients)
print(measures, type(measures))

def safeParse(val):
    if re.match(r'^-?\d+(?:\.\d+)?$', val) is None:
        return 1
    return float(val)

def summAmoutOfIngredient(sources, selected_ingr, measures):
    amount = 0
    for line in sources:
        words = re.split('\s', line)
        if len(words)>2:
            ingr = ' '.join(words[2::]).lstrip().rstrip()
            if selected_ingr == ingr and len(words[0])>0 and len(words[1])>0 and words[1] in measures:
                amount = amount + measures.get(words[1])*safeParse(words[0])
    return amount

ingr_data = dict()
for i, ingr in enumerate(ingredients):
    values = []
    for recipe in df['Ingredients']:
        values.append(summAmoutOfIngredient(recipe.split(', '), ingr, measures))
    df[ingr] = values
print(df.shape)

df = df.drop(['Ingredients'], axis=1)
print(df.columns)
print(df.head())

df = df.loc[:, (df != 0).any(axis=0)] # удаление столбцов с нулями
print(df.head())
print(df.shape)

df.to_csv(curr_dir +'/datasets/dataset_step_1.csv', index=False)
# END OF DATASET №1 ==================================================

main_df = df
used_ingredients = list(df.columns)
used_ingredients = used_ingredients[3::]

# DATASET №2 ==================================================
print('>>> PROCESSING DATASET #2')
df = pd.read_csv(curr_dir + '/datasets/all_drinks.csv')
print(df.shape)
print(df.sample(50))
print(df.columns)
print(used_ingredients)
df = df.fillna('')
df = formatMeasureValues(df, replacing_dict)

new_measures = set()
new_ingredients = set()
total_ingredients = set()

def prepareMeasure(val):
    val = str(val).strip().lower()
    if len(val)>0 and val!='nan':
        if not any(m in val for m in measures):
            new_measures.add(val)

def prepareIngredient(val):
    val = str(val).strip().lower()
    if len(val)>0 and val!='nan':
        total_ingredients.add(val)
        if not val in used_ingredients:
            new_ingredients.add(val)

for ind, row in df.iterrows():
    for i in range(1,16):
        prepareMeasure(row['strMeasure' + str(i)])
        prepareIngredient(row['strIngredient' + str(i)])

print(len(new_measures), new_measures)
print(len(new_ingredients), new_ingredients)
print(len(total_ingredients), total_ingredients)

for ingr in new_ingredients:
    zeros = [0 for x in range(main_df.shape[0])]
    main_df[ingr] = zeros

used_ingredients.extend(list(new_ingredients))
print(main_df.shape)
main_df.sample(50)

print(len(used_ingredients), used_ingredients)

print(main_df.shape)

def calculateValueOfIngredient(measureValue, measures):
    for m in measures:
        if m in measureValue:
            val = measureValue.replace(m, '').strip()
            val = getNumber(val)
            if val > 0:
                return val*measures[m]
    return 1

zeros = [0 for x in range(main_df.shape[1] - 1)]
for ind, row in df.iterrows():
    row_list = [row['strDrink']]
    row_list.extend(zeros)
    for i in range(1,16):
        measureValue = str(row['strMeasure' + str(i)]).strip().lower()
        ingredientName = str(row['strIngredient' + str(i)]).strip().lower()
        if len(ingredientName)>0:
            row_list[main_df.columns.get_loc(ingredientName)] = calculateValueOfIngredient(measureValue, measures)
    row_df = pd.DataFrame([row_list], columns=list(main_df.columns))
    main_df = main_df.append(row_df, ignore_index=True)
print(main_df.shape)

print(main_df.sample(100))

main_df = main_df.loc[:, (main_df != 0).any(axis=0)]
print(main_df.head())
print(main_df.shape)

main_df.to_csv(curr_dir +'/datasets/dataset_step_2.csv', index=False)
# END OF DATASET №2 ==================================================


# DATASET №3 ==================================================
print('>>> PROCESSING DATASET #3')
df = pd.read_csv(curr_dir + '/datasets/mr-boston-flattened.csv')
print(df.shape)
print(df.sample(50))
print(df.columns)

df = df.fillna('')
df = formatMeasureValues(df, replacing_dict)
print(df.sample(50))

new_measures = set()
new_ingredients = set()
total_ingredients = set()

def prepareMeasure(val):
    val = str(val).strip().lower()
    if len(val)>0 and val!='nan':
        if not any(m in val for m in measures):
            new_measures.add(val)

def prepareIngredient(val):
    val = str(val).strip().lower()
    if len(val)>0 and val!='nan':
        total_ingredients.add(val)
        if not val in used_ingredients:
            new_ingredients.add(val)

for ind, row in df.iterrows():
    for i in range(1,7):
        prepareMeasure(row['measurement-' + str(i)])
        prepareIngredient(row['ingredient-' + str(i)])

print(len(new_measures), new_measures)
print(len(new_ingredients), new_ingredients)
print(len(total_ingredients), total_ingredients)

for ingr in new_ingredients:
    zeros = [0 for x in range(main_df.shape[0])]
    main_df[ingr] = zeros

used_ingredients.extend(list(new_ingredients))
print(main_df.shape)
print(main_df.sample(50))

print(main_df.shape)

zeros = [0 for x in range(main_df.shape[1] - 1)]
for ind, row in df.iterrows():
    row_list = [row['name']]
    row_list.extend(zeros)
    for i in range(1,7):
        measureValue = str(row['measurement-' + str(i)]).strip().lower()
        ingredientName = str(row['ingredient-' + str(i)]).strip().lower()
        if len(ingredientName)>0:
            row_list[main_df.columns.get_loc(ingredientName)] = calculateValueOfIngredient(measureValue, measures)
    row_df = pd.DataFrame([row_list], columns=list(main_df.columns))
    main_df = main_df.append(row_df, ignore_index=True)
print(main_df.shape)

main_df = main_df.loc[:, (main_df != 0).any(axis=0)]
print(main_df.shape)

main_df.to_csv(curr_dir +'/datasets/dataset_step_3.csv', index=False)
# END OF DATASET №3 ==================================================
print('>>> done')
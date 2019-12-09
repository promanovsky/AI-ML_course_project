import pandas as pd
import os
import re

"""
Этап 4b 
Анализ ингредиентов - поиск вхождений коротких описаний в длинные, мерж столбцов,
анализ количественного состава колонок - удаляем колонки в которых только одно значение и которые никуда не входят 
по наименованию.
"""

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

pd.set_option('display.max_columns', 1856)
pd.set_option('display.max_rows', 50)

par_dir = os.path.abspath(os.pardir)
df = pd.read_csv(par_dir + '/datasets/columns_editing.csv')
#df = pd.read_csv(io.StringIO(df.to_csv(index=False)))
print(df.shape)
df.sample(10)

numerical_columns = [c for c in df.columns if df[c].dtype.name != 'object']
non_numerical_columns = [c for c in df.columns if df[c].dtype.name == 'object']
print(len(numerical_columns), numerical_columns)
print(len(non_numerical_columns), non_numerical_columns)

weak_ingredients = set()
for c in numerical_columns:
    values = list(df[c].unique())
    count_values = list(df[c].value_counts())
    if values[0] == 0 and count_values[1] == 1 and len(values) <=2:
        weak_ingredients.add(c)

print(len(weak_ingredients), weak_ingredients)

short_columns = set()
for c in numerical_columns:
    if len(c.split(' '))<=2 and not c in weak_ingredients and c!='rating':
        short_columns.add(c)
short_columns = sorted(short_columns, key=len, reverse=True)
print(len(short_columns),short_columns)

def check_words_in(short, long):
    words = short.split(' ')
    result = True
    for word in words:
        result = (not findWholeWord(word)(long) is None) and result
    return result

ingr_to_delete = set()
drop_columns = set(weak_ingredients)
for weak in weak_ingredients:
    for short in short_columns:
        if check_words_in(short, weak):
            df[short] = df[short] + df[weak]
            #print(weak, 'CONSUMED', short)
            ingr_to_delete.add(weak)
            drop_columns.remove(weak)
            break

print(len(ingr_to_delete), ingr_to_delete)

df = df.drop(ingr_to_delete, axis=1)
df = df.drop(drop_columns, axis=1)
df = df[(df.T != 0).any()]

print(df.shape)
print(list(df.columns))

df.to_csv(par_dir +'/datasets/reducing_dataset.csv', index=False)
print('done')
# (4326, 792)
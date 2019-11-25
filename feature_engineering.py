import pandas as pd
import os

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + '/datasets/final_dataset.csv')
print(df.shape)
#print(df.sample(20))

used_ingredients = set(list(df.columns)[2::])
print('used_ingredients', len(used_ingredients), used_ingredients)
used_ingredients.remove('rating')

short_ingredients = set()
MIN_SIZE = 2
MAX_SIZE = 3

for ingr in used_ingredients:
    size = len(ingr.split(' '))
    if MIN_SIZE <= size <= MAX_SIZE:
        short_ingredients.add(ingr)

def ingr_full_contain(ing, long_ingr):
    if ing == long_ingr or len(long_ingr) < len(ing):
        return False
    words = ing.split(' ')
    result = True
    for word in words:
        result = word in long_ingr and result
    #if result:
    #   print(long_ingr, 'CONTAINS', ing)
    return result

short_ingredients = sorted(short_ingredients, key=len, reverse=False)
print('short_ingredients', len(short_ingredients), short_ingredients)

del_ingredients = set()
for ingr in used_ingredients:
    for short in short_ingredients:
        if ingr_full_contain(short, ingr):
            df[short] = df[short] + df[ingr]
            del_ingredients.add(ingr)
            break

print('del_ingredients', len(del_ingredients), del_ingredients)
print('Shape before deleting',df.shape)
df = df.drop(del_ingredients, axis=1)
print('Shape after deleting',df.shape)
df.to_csv(curr_dir +'/datasets/engineering_in_progress.csv')
print('done')
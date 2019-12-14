from sklearn.externals import joblib
from common.tools import ingredients_transformation, find_group_for_ingredient, findMeasure, measures
import numpy as np

loaded_model = joblib.load('random_forest_classifier_model.mdl')
lenc =  joblib.load('random_forest_label_encoder.lenc')

print(type(loaded_model), loaded_model)
print(len(ingredients_transformation))

param1 = [('rosé wine', 1, 'bottle'), ('strawberry liqueur', 0.5, 'bottle'), ('nutmeg', 1, 'cup')]
param2 = [('apricot brandy', 25, 'shot'), ('coca-cola', 1, 'full glass'), ('cranberry vodka', 5, 'bottle')]
param3 = [('red wine', .02, 'l'), ('plymouth gin', .022, 'l'), ('crystallised ginger', 1, 'tsp'), ('cloves', 2, 'tsp'), ('tonic', 0.5, 'l')]

def prepare_data(param):
    data = [0 for x in range(len(ingredients_transformation))]
    for item in param:
        ingr_group, index = find_group_for_ingredient(item[0])
        value = measures[findMeasure(item[2], measures)] * item[1]
        data[index] = value
    return np.asarray(data).reshape(1,-1)

print('prediction =', lenc.inverse_transform(loaded_model.predict(prepare_data(param1))))
print('prediction =', lenc.inverse_transform(loaded_model.predict(prepare_data(param2))))
print('prediction =', lenc.inverse_transform(loaded_model.predict(prepare_data(param3))))

"""
wine_ingr
scotch_ingr
whisky_ingr
liquer_ingr
pepper_ingr
syrup_ingr
soda_ingr
vermouth_ingr
cinnamon_ingr
brandy_ingr
orange_ingr
absinthe_ingr
vodka_ingr
sherry_ingr
beer_ingr
gin_ingr
lemonade_ingr
bitter_ingr
rum_ingr
water_ingr
milk_ingr
tequila_ingr
olive_ingr
juice_ingr
coffee_ingr
lemon_ingr
egg_ingr
tea_ingr
cider_ingr
grapefruit_ingr
ginger_ingr
butter_ingr
cream_ingr
sauce_ingr
cloves_ingr
cucumber_ingr
common_ingr
"""
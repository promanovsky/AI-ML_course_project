from sklearn.externals import joblib

from common.tools import ingredients_transformation, prepare_data

loaded_model = joblib.load('random_forest_classifier_model.mdl')
lenc =  joblib.load('random_forest_label_encoder.lenc')
scaler = joblib.load('random_forest_scaler.sclr')

print(type(loaded_model), loaded_model)
print(len(ingredients_transformation))

param1 = [('rosé wine', 1, 'bottle'), ('strawberry liqueur', 0.5, 'bottle'), ('nutmeg', 1, 'cup')]
param2 = [('apricot brandy', 25, 'shot'), ('coca-cola', 1, 'full glass'), ('cranberry vodka', 5, 'bottle')]
param3 = [('red wine', 2, 'l'), ('plymouth gin', 1.5, 'l'), ('crystallised ginger', 50, 'tsp'), ('cloves', 20, 'tsp'), ('tonic', 0.5, 'l')]

print('prediction =', lenc.inverse_transform(loaded_model.predict(prepare_data(param1, scaler))))
print('prediction =', lenc.inverse_transform(loaded_model.predict(prepare_data(param2, scaler))))
print('prediction =', lenc.inverse_transform(loaded_model.predict(prepare_data(param3, scaler))))

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
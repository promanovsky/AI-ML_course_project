from sklearn.externals import joblib
from common.tools import ingredients_transformation, find_group_for_ingredient, findMeasure, measures
import numpy as np

loaded_model = joblib.load('random_forest_classifier_model.mdl')
lenc =  joblib.load('random_forest_label_encoder.lenc')

print(type(loaded_model), loaded_model)
print(len(ingredients_transformation))

param1 = [('rosé wine', 1, 'bottle'), ('strawberry liqueur', 0.5, 'bottle'), ('nutmeg', 1, 'cup')]
param2 = [('apricot brandy', 25, 'shot'), ('coca-cola', 1, 'full glass'), ('cranberry vodka', 5, 'bottle')]

def prepare_data(param):
    data = [0 for x in range(len(ingredients_transformation))]
    for item in param:
        ingr_group, index = find_group_for_ingredient(item[0])
        value = measures[findMeasure(item[2], measures)] * item[1]
        data[index] = value
    return np.asarray(data).reshape(1,-1)

pred = loaded_model.predict(prepare_data(param1))
pred = lenc.inverse_transform(pred)
print('prediction =', pred)

pred = loaded_model.predict(prepare_data(param2))
pred = lenc.inverse_transform(pred)
print('prediction =', pred)
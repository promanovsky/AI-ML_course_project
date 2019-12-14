import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

dataset_path = '/datasets/grouped_columns_filled_with_regression.csv'

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + dataset_path)
print(df.shape)

columns_to_scale = list(df.columns)
columns_to_scale.remove('Cocktail Name')
columns_to_scale.remove('rating')

scaler = StandardScaler()
scaler.fit(df[columns_to_scale])
scaled_data = scaler.fit_transform(df[columns_to_scale])

lenc = LabelEncoder()
Y = lenc.fit_transform(df['rating'])
X_Train, X_Test, Y_Train, Y_Test = train_test_split(scaled_data, Y, test_size = 0.15, random_state = 42)
trainedforest = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.2, min_samples_leaf=6, min_samples_split=8, n_estimators=100).fit(X_Train,Y_Train)
print('model score =', trainedforest.score(X_Test, Y_Test))

joblib.dump(trainedforest, 'random_forest_classifier_model.mdl')
joblib.dump(lenc, 'random_forest_label_encoder.lenc')
joblib.dump(scaler, 'random_forest_scaler.sclr')
print('done')
# model score = 0.5485362095531587
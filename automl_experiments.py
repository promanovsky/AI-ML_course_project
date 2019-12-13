import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

sns.set(font_scale=1.5, palette="colorblind")

dataset_path = "/datasets/grouped_columns_filled_with_regression.csv"

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + dataset_path)
print(df.shape)

columns_to_scale = list(df.columns)
columns_to_scale.remove('Cocktail Name')
columns_to_scale.remove('rating')

scaler = StandardScaler()
scaler.fit(df[columns_to_scale])
standart_scaled = scaler.fit_transform(df[columns_to_scale])

X = standart_scaled

Y = df['rating']
y = LabelEncoder().fit_transform(Y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(verbosity=2,
                      scoring="accuracy",
                      random_state=42,
                      periodic_checkpoint_folder="tpot_temporary_results",
                      n_jobs=-1,
                      generations=2,
                      population_size=20)
times = []
scores = []
winning_pipes = []

RUNS = 10
for x in range(RUNS):
    start_time = timeit.default_timer()
    tpot.fit(X_train, y_train)
    elapsed = timeit.default_timer() - start_time
    times.append(elapsed)
    winning_pipes.append(tpot.fitted_pipeline_)
    scores.append(tpot.score(X_test, y_test))
    tpot.export('tpot_pipelines_export.py')

times = [time/60 for time in times]
print('Times:', times)
print('Scores:', scores)
print('Winning pipelines:', winning_pipes)

timeo = np.array(times)
df = pd.DataFrame(np.reshape(timeo, (len(timeo))))
df= df.rename(columns={0: "Times"})
df = df.reset_index()
df = df.rename(columns = {"index": "Runs"})
print(df)

ax = sns.barplot(x= np.arange(1, 11), y = "Times", data = df)
ax.set(xlabel='Run # for Set of {} Pipelines'.format(RUNS), ylabel='Time in Minutes')
plt.title("TPOT Run Times")
plt.show()
import pandas as pd
import os
import pandas_profiling

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + '/datasets/engineering_in_progress.csv')
print(df.shape)

"""
pip install alembic
pip install missingno
pip install xgboost
pip install pandas_profiling
Датасет содержит строго числовые данные (без категориальных)
"""
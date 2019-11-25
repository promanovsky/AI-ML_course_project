import pandas as pd
import os

curr_dir = os.path.abspath(os.curdir)
df = pd.read_csv(curr_dir + '/datasets/engineering_in_progress.csv')
print(df.shape)
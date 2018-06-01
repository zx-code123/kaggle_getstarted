#%%
import os
import pandas as pd
import numpy as np
dataset_dir = "digit_data"
data_train = pd.read_csv(os.path.join(dataset_dir,"train.csv"))
train_df = data_train.values[0:,1:]
print(train_df)
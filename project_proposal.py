import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import glob
import os

path = r'data/'

csv_files = ["YVRStationData_2013-2015.csv", "YVRStationData_2015-2017.csv",
             "YVRStationData_2017-2019.csv", "YVRStationData_2019-2021.csv", "YVRStationData_2021-2023.csv"]

df = pd.DataFrame()

for file in csv_files:
    df_temp = pd.read_csv(path+file)
    df = df.append(df_temp, ignore_index=True)

df.columns = df.columns.str.strip()

df["TEMP_MEAN"].head()

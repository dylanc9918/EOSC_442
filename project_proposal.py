import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
import glob
import os

path_to_folder = r'data/'

csv_files = ["YVRStationData_2013-2015.csv", "YVRStationData_2015-2017.csv",
             "YVRStationData_2017-2019.csv", "YVRStationData_2019-2021.csv", "YVRStationData_2021-2023.csv"]

df = pd.DataFrame()

# read in data files separatly and append to eachother
for file in csv_files:
    df_temp = pd.read_csv(path_to_folder+file, header=[0, 1])
    df = df.append(df_temp, ignore_index=True)

# join the gases and their units together via an underscore
df.columns = df.columns.map('_'.join)

# replace whitespace within columns
df.columns = df.columns.str.replace(' ', '')

# renamed to date to make it easier to index
df = df.rename(columns={"DateTime_": "date"})

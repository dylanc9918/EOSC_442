import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from datetime import datetime
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


# Formats the date column from a string to datetime column in pandas
df['date'] = pd.to_datetime(df['date'], format=" %m/%d/%Y %H%M:%S %p ")

plt.scatter(df["date"], df["CO_ppm"])
dtFmt = mdates.DateFormatter('%Y')  # define the formatting
# apply the format to the desired axis
plt.gca().xaxis.set_major_formatter(dtFmt)
plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))

plt.ylabel("Concentration (ppm)")
plt.xlabel("Date")
plt.show()

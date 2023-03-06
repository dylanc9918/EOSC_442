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

# change all columns to numeric except for date
cols = [i for i in df.columns if i not in ["date"]]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Remove empty Station column
df = df.drop(columns=["Station:VancouverInternationalAirport#2-_"])


# Formats the date column from a string to datetime column in pandas
df['date'] = pd.to_datetime(df['date'], format=" %m/%d/%Y %H%M:%S %p ")


# creates a plot of the monthly averages for the CO_ppm column
CO_monthly_avg = df.groupby(pd.PeriodIndex(
    df['date'], freq="M"))['CO_ppm'].mean()
CO_monthly_avg.plot()


fig, ax = plt.subplots(3, 2, sharex=True)
fig.tight_layout(h_pad=2)


ax[0, 0].CO_monthly_avg.plot()
ax[0, 1].plot(df['date'], df["NO_ppb"])
ax[1, 0].plot(df['date'], df["NO2_ppb"])
ax[1, 1].plot(df['date'], df["O3_ppb"])
ax[2, 0].plot(df['date'], df["SO2_ppb"])
ax[2, 1].plot(df['date'], df["PM10_ug/m3"])


ax[0, 0].set_title("CO Levels")
ax[0, 1].set_title("NO Levels")
ax[1, 0].set_title("NO2 Levels")
ax[1, 1].set_title("O3 Levels")
ax[2, 0].set_title("SO2 Levels")
ax[2, 1].set_title("PM10_ug/m3 Levels")


dtFmt = mdates.DateFormatter('%Y')  # define the formatting
# apply the format to the desired axis
plt.gca().xaxis.set_major_formatter(dtFmt)


plt.ylabel("Concentration (ppm)")
plt.xlabel("Date")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from datetime import datetime
import scipy
import numpy as np
import glob
import os

path_to_folder = r'data/'

csv_files = [
    # "YVRStationData_1998-1999.csv", "YVRStationData_1999-2001.csv",
    #          "YVRStationData_2001-2003.csv",
    #          "YVRStationData_2003-2005.csv", "YVRStationData_2005-2007.csv", "YVRStationData_2007-2009.csv",
    #          "YVRStationData_2009-2011.csv",
    # "YVRStationData_2011-2013csv",
    "YVRStationData_2013-2015.csv",
    "YVRStationData_2015-2017.csv",
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

# renamed to columns to make it easier to index
df = df.rename(columns={"DateTime_": "date"})
df = df.rename(columns={"TEMP_MEAN_°C": "temp"})
df = df.rename(columns={"PM25_ug/m3": "PM25"})
df = df.rename(columns={"PM10_ug/m3": "PM10"})


# change all columns to numeric except for date
cols = [i for i in df.columns if i not in ["date"]]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Remove empty Station column and other unecessary columns
df = df.drop(columns=["Station:VancouverInternationalAirport#2-_", "WSPD_SCLR_m/s", 'WDIR_SCLR_Deg.',
             'WSPD_VECT_m/s', 'WDIR_VECT_Deg.', 'ATM_PRESS_1HR_kPa', 'HUMIDITY_%RH', 'RAD_TOTAL_W/M**2', "NOx_ppb", 'SO2_ppb', 'NO_ppb'])


# Formats the date column from a string to datetime column in pandas
df['date'] = pd.to_datetime(df['date'], format=" %m/%d/%Y %H%M:%S %p ")


# creates a plot of the monthly averages for the all columns
monthly_avg = df.groupby(pd.PeriodIndex(
    df['date'], freq="M")).mean()


fig1, ax1 = plt.subplots(len(monthly_avg.columns),
                         figsize=[25, 15], sharex=True)
fig1.tight_layout()

titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]

units = ["ppm", "ppb", "ppb", "µg/m³", "µg/m³", "Celsius", "mm"]

colors = sns.color_palette(None, len(monthly_avg.columns))

# loop through each column of the dataframe
for i in range(len(monthly_avg.columns)):
    # activate the subplot

    # plot the time-series
    ax1[i].plot(monthly_avg.index.timestamp(),
                monthly_avg.iloc[:, i], color=colors[i])

    # add figure elements
    ax1[i].set_title(titles[i])
    ax1[i].set_ylabel(units[i])
    plt.xlabel("Date")


year_avg = df.groupby(pd.PeriodIndex(
    df['date'], freq="Y")).mean()


# creates plots of Yearly averages
fig, ax = plt.subplots(len(year_avg.columns), figsize=[25, 15], sharex=True)
fig.tight_layout()

titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]

units = ["ppm", "ppb", "ppb", "µg/m³", "µg/m³", "Celsius", "mm"]

colors = sns.color_palette(None, len(year_avg.columns))

# loop through each column of the dataframe
for i in range(len(year_avg.columns)):
    # activate the subplot

    # plot the time-series
    ax[i].plot(year_avg.index.strftime("%Y"),
               year_avg.iloc[:, i], color=colors[i])

    # add figure elements
    ax[i].set_title(titles[i])
    ax[i].set_ylabel(units[i])
    plt.xlabel("Date")


monthly_avg.plot(subplots=True, layout=(
    5, 2), grid=True, sharex=True, figsize=(15, 8))
plt.tight_layout()


# creates season mapping based on months in dataframe
season = ((df.date.dt.month % 12 + 3) //
          3).map({1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'})

# adds the seasonal mapping to the dataframe
df['season'] = season


# groups by months
seasonal_avg = df.groupby([pd.PeriodIndex(
    df['date'], freq="M"), "season"]).mean()

seasonal_avg.plot(subplots=True, grid=True, sharex=True, figsize=(15, 8))
plt.tight_layout()


# list of  gasses and their respective seasons
# 0 = Winter (DJF)
# 0 = Spring (MAM)
# 0 = Summer (JJA)
# 0 = Fall (SON)


seasonal_avg = seasonal_avg.groupby("season")

# creates list of gasses and their seasons
season_CO_list = list(seasonal_avg.CO_ppm)
season_O3_list = list(seasonal_avg.O3_ppb)
season_PM25_list = list(seasonal_avg.PM25)
season_PM10_list = list(seasonal_avg.PM10)
season_NO2_list = list(seasonal_avg.NO2_ppb)
season_temp_list = list(seasonal_avg.temp)

fig, ax = plt.subplots(4, sharex=True)
fig.tight_layout(h_pad=2)

for i in range(4):
    ax[i].scatter(season_CO_list[i][1], season_temp_list[i][1])


# fig, ax = plt.subplots(3, 2, sharex=True)
# fig.tight_layout(h_pad=2)


# ax[0, 0].plot(monthly_avg["CO_ppm"])
# ax[0, 1].plot(df['date'], df["NO_ppb"])
# ax[1, 0].plot(df['date'], df["NO2_ppb"])
# ax[1, 1].plot(df['date'], df["O3_ppb"])
# ax[2, 0].plot(df['date'], df["SO2_ppb"])
# ax[2, 1].plot(df['date'], df["PM10_ug/m3"])


# ax[0, 0].set_title("CO Levels")
# ax[0, 1].set_title("NO Levels")
# ax[1, 0].set_title("NO2 Levels")
# ax[1, 1].set_title("O3 Levels")
# ax[2, 0].set_title("SO2 Levels")
# ax[2, 1].set_title("PM10_ug/m3 Levels")


# dtFmt = mdates.DateFormatter('%Y')  # define the formatting
# # apply the format to the desired axis
# plt.gca().xaxis.set_major_formatter(dtFmt)


# plt.ylabel("Concentration (ppm)")
# plt.xlabel("Date")
# plt.show()

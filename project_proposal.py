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
from sklearn.linear_model import LinearRegression
import scipy.stats as stat
import statsmodels.api as sm

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
fig1.tight_layout(pad=5)
fig1.suptitle("Monthly Averages")

titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]

units = ["ppm", "ppb", "ppb", "µg/m³", "µg/m³", "Celsius", "mm"]

colors = sns.color_palette(None, len(monthly_avg.columns))

# loop through each column of the dataframe
for i in range(len(monthly_avg.columns)):
    # activate the subplot

    # plot the time-series
    ax1[i].plot(monthly_avg.index.to_timestamp(),
                monthly_avg.iloc[:, i], color=colors[i])

    # add figure elements
    ax1[i].set_title(titles[i])
    ax1[i].set_ylabel(units[i])
    plt.xlabel("Date")


year_avg = df.groupby(pd.PeriodIndex(
    df['date'], freq="Y")).mean()


# creates plots of Yearly averages
fig, ax = plt.subplots(len(year_avg.columns), figsize=[25, 15], sharex=True)
fig.tight_layout(pad=5)
fig.suptitle("Yearly Averages")


titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]

units = ["ppm", "ppb", "ppb", "µg/m³", "µg/m³", "Celsius", "mm"]

colors = sns.color_palette(None, len(year_avg.columns))

# loop through each column of the dataframe
for i in range(len(year_avg.columns)):
    # activate the subplot

    # plot the time-series
    ax[i].plot(year_avg.index.to_timestamp(),
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

list_gas = [season_CO_list,
            season_O3_list,
            season_PM25_list,
            season_PM10_list,
            season_NO2_list]
# creates subplots of each season and looks at their monthly distribution of temperature vs concentration for each graph. Creates a linear regression line and in addition reports the R-squared values and the p-values

list_gas = [season_CO_list,
            season_NO2_list,
            season_O3_list,
            season_PM25_list,
            season_PM10_list]


titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]

for j in range(len(list_gas)):
    fig.text(0.5, 0.00, "Units", ha='center')
    fig.text(0.00, 0.5, 'Temp (C°)', va='center', rotation='vertical')
    season = ["Winter", "Spring", "Summer", "Fall"]
    fig, ax = plt.subplots(4)
    fig.tight_layout(h_pad=2)
    fig.suptitle("Seasonal Correlation Graph between " +
                 titles[j] + " and Temperature", y=1.05)

    for i in range(4):
        ax[i].scatter(list_gas[j][i][1], season_temp_list[i][1])
        ax[i].set_title(season[i])
        model = sm.OLS(season_temp_list[i][1],
                       list_gas[j][i][1], missing='drop').fit()
        y_predict = model.predict(list_gas[j][i][1])
        rmse = sm.tools.eval_measures.rmse(season_temp_list[i][1], y_predict)

        text = 'R-Squared:{:.4f} \np-Value:{:.2E}'.format(
            model.rsquared, model.pvalues[0])
        ax[i].plot(list_gas[j][i][1], y_predict,
                   linestyle="-", color="r")
        ax[i].annotate(text, xy=(0, 1.1),
                       xycoords='axes fraction', fontsize=8, color='r')

    fig.text(0.5, 0.00, "CO Concentration ppm", ha='center')
    fig.text(0.00, 0.5, 'Temp (C°)', va='center', rotation='vertical')
    season = ["Winter", "Spring", "Summer", "Fall"]


for i in range(4):
    ax[i].scatter(season_CO_list[i][1], season_temp_list[i][1])
    ax[i].set_title(season[i])
    model = sm.OLS(season_temp_list[i][1],
                   season_CO_list[i][1], missing='drop').fit()
    y_predict = model.predict(season_CO_list[i][1])
    rmse = sm.tools.eval_measures.rmse(season_temp_list[i][1], y_predict)

    text = 'R-Squared:{:.4f} \np-Value:{:.2E}'.format(
        co_model.rsquared, co_model.pvalues[0])
    ax[i].plot(season_CO_list[i][1], y_predict,
               linestyle="-", color="r")
    ax[i].annotate(text, xy=(0, 1.2), xycoords='axes fraction',
                   fontsize=8, color='r')

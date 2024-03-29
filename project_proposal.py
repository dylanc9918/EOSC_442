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
# monthly_avg = df.groupby(pd.PeriodIndex(
#     df['date'], freq="M")).mean()


# fig1, ax1 = plt.subplots(len(monthly_avg.columns),
#                          figsize=[25, 15], sharex=True)
# fig1.tight_layout(pad=5)
# fig1.suptitle("Monthly Averages")

# titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
#           "Temperature", "Precipitation Total"]

# units = ["ppm", "ppb", "ppb", "µg/m³", "µg/m³", "Celsius", "mm"]

# colors = sns.color_palette(None, len(monthly_avg.columns))

# # loop through each column of the dataframe
# for i in range(len(monthly_avg.columns)):
#     # activate the subplot

#     # plot the time-series
#     ax1[i].plot(monthly_avg.index.to_timestamp(),
#                 monthly_avg.iloc[:, i], color=colors[i])

#     # add figure elements
#     ax1[i].set_title(titles[i])
#     ax1[i].set_ylabel(units[i])
#     plt.xlabel("Date")


# year_avg = df.groupby(pd.PeriodIndex(
#     df['date'], freq="Y")).mean()


# creates plots of Yearly averages
# fig, ax = plt.subplots(len(year_avg.columns), figsize=[25, 15], sharex=True)
# fig.tight_layout(pad=5)
# fig.suptitle("Yearly Averages")


# titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
#           "Temperature", "Precipitation Total"]

# units = ["ppm", "ppb", "ppb", "µg/m³", "µg/m³", "Celsius", "mm"]

# colors = sns.color_palette(None, len(year_avg.columns))

# # loop through each column of the dataframe
# for i in range(len(year_avg.columns)):
#     # activate the subplot

#     # plot the time-series
#     ax[i].plot(year_avg.index.to_timestamp(),
#                year_avg.iloc[:, i], color=colors[i])

#     # add figure elements
#     ax[i].set_title(titles[i])
#     ax[i].set_ylabel(units[i])
#     plt.xlabel("Date")


# monthly_avg.plot(subplots=True, layout=(
#     5, 2), grid=True, sharex=True, figsize=(15, 8))
# plt.tight_layout()


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


seasonal_avg = seasonal_avg.groupby("season")

# creates list of gasses and their seasons
season_CO_list = list(seasonal_avg.CO_ppm)
season_O3_list = list(seasonal_avg.O3_ppb)
season_PM25_list = list(seasonal_avg.PM25)
season_PM10_list = list(seasonal_avg.PM10)
season_NO2_list = list(seasonal_avg.NO2_ppb)
season_temp_list = list(seasonal_avg.temp)
season_precip_list = list(seasonal_avg.PRECIP_TOTAL_mm)

# reverse order of indices 1 and 2 since they are swithced out of order of winter summer spring
season_CO_list[1], season_CO_list[2] = season_CO_list[2], season_CO_list[1]

season_O3_list[1], season_O3_list[2] = season_O3_list[2], season_O3_list[1]

season_PM25_list[1], season_PM25_list[2] = season_PM25_list[2], season_PM25_list[1]

season_PM10_list[1], season_PM10_list[2] = season_PM10_list[2], season_PM10_list[1]

season_NO2_list[1], season_NO2_list[2] = season_NO2_list[2], season_NO2_list[1]

season_temp_list[1], season_temp_list[2] = season_temp_list[2], season_temp_list[1]

season_precip_list[1], season_precip_list[2] = season_precip_list[2], season_precip_list[1]


# creates subplots of each season and looks at their monthly distribution of temperature vs concentration for each graph. Creates a linear regression line and in addition reports the R-squared values and the p-values

list_gas = [season_CO_list,
            season_NO2_list,
            season_O3_list,
            ]


titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]

units = ["ppm", "ppb", "ppb"]

for j in range(len(list_gas)):
    fig, ax = plt.subplots(4, figsize=(8, 15))
    fig.text(0.00, 0.5, 'Temp (C°)', va='center', rotation='vertical')
    season = ["Winter", "Spring", "Summer", "Fall"]
    fig.tight_layout(h_pad=2)
    fig.suptitle("Seasonal Correlation Graph between " +
                 titles[j] + " and Temperature", y=1.05)

    fig.text(0.5, 0.00, units[j], ha='center')
    fig.text(0.00, 0.5, 'Temp (C°)', va='center', rotation='vertical')
    season = ["Winter", "Spring", "Summer", "Fall"]

    for i in range(4):
        season_list = list(list_gas[j][i][1])
        ax[i].scatter(season_list, season_temp_list[i][1])
        ax[i].set_title(season[i])
        season_list = sm.add_constant(season_list)
        model = sm.OLS(season_temp_list[i][1],
                       season_list, missing='drop').fit()
        y_predict = model.predict(season_list)

        text = 'R-Squared:{:.4f} \np-Value:{:.2E}'.format(
            model.rsquared, model.pvalues[0])
        ax[i].plot(season_list[:, 1], y_predict,
                   linestyle="-", color="r")
        ax[i].annotate(text, xy=(0.01, .9),
                       xycoords='axes fraction', fontsize=8, color='r')


# Creates subplots perfrom linear regression on all of the data for gases and Temperature

gas_name = ["CO", "NO2", "O3"]

titles = ["CO", "NO2", "O3", "PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]
fig, ax = plt.subplots(len(list_gas))
fig.tight_layout(h_pad=2)
fig.text(0.5, 0.00, "Units", ha='center')
fig.text(0.00, 0.5, 'Temp (C°)', va='center', rotation='vertical')


for j in range(len(list_gas)):
    fig.suptitle(
        "Correlation Graph between gas and Temperature Across all Years", y=1.05)

    gas_year_list = list(df.iloc[:, j+1])
    ax[j].scatter(gas_year_list, df.temp)
    gas_year_list = sm.add_constant(gas_year_list)
    ax[j].set_title(gas_name[j])
    model = sm.OLS(df.temp,
                   gas_year_list, missing='drop').fit()
    y_predict = model.predict(gas_year_list)

    text = 'R-Squared:{:.4f} \np-Value:{:.2E}'.format(
        model.rsquared, model.pvalues[0])
    ax[j].plot(gas_year_list[:, 1], y_predict,
               linestyle="-", color="r")
    ax[j].annotate(text, xy=(0, 1.1),
                   xycoords='axes fraction', fontsize=8, color='r')


# Creates subplots perfrom linear regression on all of the data for Particulate matter and precip

part_name = ["PM25", "PM10"]
part_mat_list = [season_PM25_list, season_PM10_list]

titles = ["PM 2.5", "PM 10"]
fig, ax = plt.subplots(len(part_mat_list))
fig.tight_layout(h_pad=2)

fig.text(0.5, 0.00, "ug/m3", ha='center')
fig.text(0.00, 0.5, 'Precip (mm)', va='center', rotation='vertical')


for j in range(len(part_mat_list)):
    fig.suptitle(
        "Correlation Graph between Particulate Matter and Precipation Across all Years", y=1.05)

    part_year_list = list(df.loc[:, part_name[j]])

    ax[j].scatter(part_year_list, df.PRECIP_TOTAL_mm)
    part_year_list = sm.add_constant(part_year_list)
    ax[j].set_title(titles[j])

    model = sm.OLS(df.PRECIP_TOTAL_mm,
                   part_year_list, missing='drop').fit()
    y_predict_part = model.predict(part_year_list)

    text = 'R-Squared:{:.4f} \np-Value:{:.2E}'.format(
        model.rsquared, model.pvalues[0])

    ax[j].plot(part_year_list[:, 1], y_predict,
               linestyle="-", color="r")

    ax[j].annotate(text, xy=(0, 1.1),
                   xycoords='axes fraction', fontsize=8, color='r')


# Creates subplots that look at the entire year of data by each year
gas_name = ["PM25", "PM10"]

titles = ["PM 2.5", "PM 10",
          "Temperature", "Precipitation Total"]

list_yrs = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]

df_group = df.groupby(df.date.dt.year)
fig.tight_layout(h_pad=2)
fig.text(0.5, 0.00, "Units", ha='center')
fig.text(0.00, 0.5, 'Precip (mm)', va='center', rotation='vertical')

for j in range(len(gas_name)):
    fig.suptitle("Correlation Graph between " +
                 gas_name[j] + " and Precipitaion for Each Year", y=.9)
    fig, ax = plt.subplots(len(list_yrs), figsize=(10, 15))

    for i in range(len(list_yrs)):
        ax[i].scatter(df_group.get_group(list_yrs[i]).loc[:, gas_name[j]],
                      df_group.get_group(list_yrs[i]).PRECIP_TOTAL_mm)
        ax[i].set_title(str(list_yrs[i]))
        model = sm.OLS(df_group.get_group(list_yrs[i]).PRECIP_TOTAL_mm,
                       df_group.get_group(list_yrs[i]).loc[:, gas_name[j]], missing='drop').fit()
        y_predict = model.predict(df_group.get_group(
            list_yrs[i]).loc[:, gas_name[j]])

        text = 'R-Squared:{:.4f} \np-Value:{:.2E}'.format(
            model.rsquared, model.pvalues[0])
        ax[i].plot(df_group.get_group(list_yrs[i]).loc[:, gas_name[j]], y_predict,
                   linestyle="-", color="r")
        ax[i].annotate(text, xy=(.8, .8),
                       xycoords='axes fraction', fontsize=8, color='r')

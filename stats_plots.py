import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import operations_model as om
import loader as l
import earnings_plots as ep
import importlib
importlib.reload(om)
importlib.reload(l)
importlib.reload(ep)

from geopy import distance
sns.set_theme()


def plot_grid_tax():
    year = list(range(2011,2024))
    input = [41, 42, 45, 45, 45, 45, 45, 49, 57, 69, 70, 71, 28]
    output = [31, 46, 51, 51, 51, 55, 55, 61, 71, 86, 91, 90, 36]
    loss = [5, 5, 5, 6, 6, 5, 5, 5.5, 5.8, 5.7, 5.8, 6, 6.5]
    tarrif = [19, 20, 18.2, 18.96, 18.24, 13.8, 10.96, 12.1] #t.o.m. 2018
    gen_tax = {year[i]: input[i] for i in range(len(year))}
    pump_tax = {year[i]: output[i] for i in range(len(year))}
    loss = {year[i]: loss[i] for i in range(len(year))}
    plt.plot(year, input)
    plt.plot(year, output)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Year')
    ax1.plot(year, output, "o-", color='midnightblue')
    ax1.plot(year, input, "o--", color='slateblue')
    ax1.set_ylabel("SEK/kW")
    ax1.tick_params(axis='y', labelcolor='blue')
    # ax2 = ax1.twinx()
    # ax2.plot(year, loss, "o-", color='indianred')
    # ax2.tick_params(axis='y', labelcolor='indianred')
    # ax2.set_ylabel("Loss [%]")
    ax1.legend(["Output", "Input"])
    ax1.set_ylim(0, 100)
    # ax2.legend(["Effect loss"], loc=9)

    fig.tight_layout()
    # plt.show()
    plt.savefig("figures/grid_costs.eps")
    plt.close()

def hours_12_21():
    df = l.load_plot_df([12, 21])
    sns.set(rc={'figure.figsize':(9,4)})
    ax = sns.violinplot(data=df, x="Hour", y="SE2", showfliers = False, inner="quartiles", scale="width", hue = 'Year', split=True)
    plt.ylim(-150, 1000)
    plt.tight_layout()
    plt.ylabel("Price [SEK/MWh]")
    plt.savefig("figures/hours_12_21_SE2.eps")
    plt.show()

def hours_some_years():
    years = [12, 15, 18, 21]
    df = l.load_plot_df(years)
    # df = df.melt(value_vars=['SE1', 'SE4'], id_vars=["Weather average", 'Hour'], value_name='Price', var_name='Region')
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lineplot(data=df, x="Hour", y="SE2", hue= 'Year', style='Year', markers=True, estimator='mean', err_style = 'band')
    # ax = sns.violinplot(data=df, x="Hour", y="SE2", showfliers = False, inner="quartiles", hue='Year', split=True)
    # plt.ylim(-150, 1000)
    plt.legend(loc='upper left')
    plt.ylabel("Price [SEK/MWh]")
    plt.xticks([0, 4, 8, 12, 16, 20, 24])
    plt.tight_layout()
    plt.savefig("figures/hours_"+str(years).replace(" ", "")+".png")
    plt.show()

def hours_regions():
    years = list(range(11,22))
    df = l.load_plot_df(years)
    df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Weather average", 'Hour'], value_name='Price', var_name='Region')
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lineplot(data=df, x="Hour", y="Price", hue= 'Region', style='Region', markers=True, estimator='mean', err_style = 'band')
    # ax = sns.violinplot(data=df, x="Hour", y="SE2", showfliers = False, inner="quartiles", hue='Year', split=True)
    # plt.ylim(-150, 1000)
    plt.ylabel("Price [SEK/MWh]")
    plt.xticks([0, 4, 8, 12, 16, 20, 24])
    plt.tight_layout()
    plt.savefig("figures/hours_11_21_regions.png")
    plt.show()

def months():
    years = [12, 15, 18, 21]
    df = l.load_plot_df(years)
    sns.set(rc={'figure.figsize':(9,4)})
    # df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Weather average", 'Month'], value_name='Price', var_name='Region')
    ax = sns.lineplot(data=df, x="Month", y="SE2", hue= 'Year', style='Year', markers=True, estimator='mean', err_style = 'band')
    # plt.ylim(-10, 90)
    plt.ylabel("Price [SEK/MWh]")
    plt.tight_layout()
    # plt.savefig("figures/month_11_21.eps")
    plt.show() 

# def Weekday_22():
#     weekdays = ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]
#     df = plot_day_price([22])
#     sns.set(rc={'figure.figsize':(7,4)})
#     ax = sns.violinplot(data=df, x="Weekday", y="Price", showfliers = False, inner="quartiles", scale="width")
#     ax.set_xticklabels(weekdays)
#     plt.ylim(-50, 250)
#     plt.tight_layout()
#     plt.ylabel("Price [SEK/MWh]")
#     plt.savefig("figures/weekdays_22.eps")
#     plt.show()

def weekday():
    weekdays = ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]
    df = plot_day_price(list(range(11,23)))
    df['Year span'] = df['Year'].apply(lambda x: "2011-2021" if x < 2022 else 2022)
    df["before"] = df["Year"] < 2022
    sns.set(rc={'figure.figsize':(9,4)})
    ax = sns.violinplot(data=df, x="Weekday", y="Price", showfliers = False, inner="quartiles", scale="width", hue="Year span", split=True)
    # ax.legend(title='Year span', loc='upper right', legend=['2022', '2011-2021'])
    ax.set_xticklabels(weekdays)
    plt.ylim(-20, 175)
    plt.tight_layout()
    plt.ylabel("Price [SEK/MWh]")
    plt.savefig("figures/weekdays_11_22.eps")
    plt.show()


def years_11_22():
    df = l.load_plot_df(list(range(11,23)))
    sns.set(rc={'figure.figsize':(9,4)})
    ax = sns.violinplot(data=df, x="Year", y="SE2", showfliers = False, inner="quartiles")
    plt.ylim(-150, 1000)
    plt.tight_layout()
    plt.ylabel("Price [SEK/MWh]")
    # plt.savefig("figures/years_11_22.eps")
    plt.show()

def years_regions_19_22():
    df = l.load_plot_df(list(range(19,23)))
    df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Weather average", 'Year'], value_name='Price', var_name='Region')
    sns.set(rc={'figure.figsize':(9,5)})
    ax = sns.violinplot(data=df, x="Year", y="Price", showfliers = False, inner="quartiles", hue="Region")
    plt.ylim(-200, 3000)
    plt.tight_layout()
    plt.ylabel("Price [SEK/MWh]")
    plt.legend(loc='upper left')
    plt.savefig("figures/years_19_22_regions.eps")
    plt.show()

def years_regions_11_22():
    df = l.load_plot_df(list(range(11,23)))
    df.rename(mapper={'Price SE1': 'SE1', 'Price SE2': 'SE2', 'Price SE3': 'SE3', 'Price SE4': 'SE4'}, axis=1, inplace=True)
    df = df.melt(value_vars=['SE1', 'SE4'], id_vars=["Weather average", 'Year'], value_name='Price', var_name='Region')
    sns.set(rc={'figure.figsize':(9,5)})
    ax = sns.violinplot(data=df, x="Year", y="Price", showfliers = False, inner="quartiles", hue="Region", split=True)
    plt.ylim(-200, 2500)
    plt.tight_layout()
    plt.ylabel("Price [SEK/MWh]")
    plt.legend(loc='upper left')
    plt.savefig("figures/years_11_22_regions14.eps")
    plt.show()

def years_variation():
    years = list(range(11,23))
    df = l.load_plot_df(years)
    stds = pd.DataFrame(columns=['Year', 'Standard deviation'])
    for y in years:
        stds.loc[len(stds.index)] = [y+2000, np.std(df[df.Year == y+2000].Price)]
        # stds.append([y+2000, np.std(df[df.Year == y+2000].Price)], ignore_index = True)
        # stds = stds + [np.std(df[df.Year == y+2000].Price)]
    sns.lmplot(data = stds, x='Year', y='Standard deviation')
    sns.set(rc={'figure.figsize':(9,4)})
    ax = sns.violinplot(data=df, x="Year", y="Price", showfliers = False, inner="quartiles")
    plt.ylim(-20, 150)
    plt.tight_layout()
    plt.ylabel("Price [SEK/MWh]")
    plt.savefig("figures/years_11_22.eps")
    plt.show()

# Capture price trends
def priceshift_years():
    day_span_back = 1
    day_span_forward = 4
    df = l.load_plot_df([19, 20, 21])
    # df["Pricechange"] = df.SE2.sub(df.SE2.shift(24))
    # df["Futureprice"] = df.SE2.sub(df.SE2.shift(-24))
    # index = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
    df["Pricechange"] = df.SE2.rolling(24*day_span_back).mean()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24*day_span_forward)
    df["Futureprice"] = df.SE2.rolling(indexer).mean()
    ord=1
    sns.lmplot(data = df, x = 'Pricechange', y = 'Futureprice', scatter_kws={'alpha':0.4}, hue='Year', order=ord, x_bins=40, height=5, aspect=1, legend=False)
    # plt.xlim([-500, 500])
    # plt.ylim([-500, 500])
    plt.legend(loc='upper left')
    plt.xlabel('Recent price change [SEK/MWh]')
    plt.ylabel('Future change [SEK/MWh]')
    plt.tight_layout()
    # plt.savefig('figures/price_trend_years_order_'+str(ord)+'.svg')
    plt.show()

# Capture price trends, varying days
def priceshift_day_span():
    df = l.load_plot_df([19, 20, 21])
    # df["Pricechange"] = df.SE2.sub(df.SE2.shift(24))
    # df["Futureprice"] = df.SE2.sub(df.SE2.shift(-24))
    day_span = 4
    try_day = list(range(1,6))
    # change_list = [('Pricechange_'+str(x)) for x in try_day]
    future_list = [('Futureprice_'+str(x)) for x in try_day]
    for d in try_day:
        df[d] = df.SE2.sub(df.SE2.rolling(24*d).mean())
        # df["Pricechange"] = df.SE2.sub(df.SE2.rolling(24).mean())
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24*day_span)
        df["Futureprice_"+str(d)] = df.SE2.sub(df.SE2.rolling(indexer).mean())
    df_change = df.melt(value_name='Pricechange', value_vars=try_day, var_name='days')
    df_future = df.melt(value_name='Futureprice', value_vars=future_list, var_name='days_future')
    comb = pd.merge(left = df_change, right = df_future, how='inner', left_index=True, right_index=True)
    comb = comb.dropna()
    # comb = df_change.merge(right=df_future, )
    # sns.set(rc={'figure.figsize':(9,5)})
    ax = sns.lmplot(data = comb, x = 'Pricechange', y = 'Futureprice', scatter_kws={'alpha':1}, hue='days', x_bins=40, legend=False, height=5, aspect=1)
    # sns.residplot(data = df, x = 'Pricechange', y = 'Futureprice', lowess=True, scatter_kws={'alpha':0.4}, order=1)
    plt.xlim([-500, 500])
    plt.ylim([-500, 500])
    plt.xlabel('Recent price change [SEK/MWh]')
    plt.ylabel('Future change [SEK/MWh]')
    plt.legend(title='Days back', loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/price_trend_days.png')
    plt.show()

    sns.set(rc={'figure.figsize':(5,5)})
    sns.residplot(data = comb[comb['days'] == 1], x = 'Pricechange', y = 'Futureprice', lowess=True, scatter_kws={'alpha':0.2}, line_kws={'color':'orange'})
    plt.xlim([-800, 800])
    plt.ylim([-800, 800])
    plt.ylabel('Residuals [SEK/MWh]')
    plt.xlabel('Future change [SEK/MWh]')
    plt.tight_layout()
    plt.savefig('figures/price_trend_days_resid.png')
    plt.show()
    return
    # df['Pricechangebin'] = pd.qcut(df['Pricechange'], q = 8)
    # # ax = sns.violinplot(data=df, x="Year", y="Price", showfliers = False, inner="quartiles", scale="width")
    # # sns.scatterplot(data=df, x="Pricechange", y = "Price")
    # # plt.show()
    # sns.violinplot(data=df, y='Price', x = 'Pricechangebin')
    # plt.ylim(-10, 120)
    # plt.show()

def price_trend():
    years = [19, 20, 21, 22]
    full_years = [i + 2000 for i in years]
    df = l.load_plot_df(years)
    day_span_back = 4
    day_span_forward = 4
    df = om.calculate_base_price_day(df, day_span_forward, 'SE2')
    df = df.rename({'Base_price': 'base_forward'}, axis=1)
    df = om.calculate_base_price_day(df, -day_span_back, 'SE2')
    df = df.rename({'Base_price': 'base_back'}, axis=1)
    
    rlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    r_table = pd.DataFrame(columns=full_years, index=rlist)
    # r_table.index = rlist
    # ans = []
    for r in rlist:
        for y in full_years:
            df_ny = df[df['Year'] == y]
            df_ny['Base_price'] = df_ny['base_back'] + r*(df_ny['base_back'] - df_ny['base_back'].shift(24))
            df_ny = df_ny.dropna()
            r_table.loc[r, y] = (sum((df_ny.base_forward - df_ny.base_back)**2) - sum((df_ny.base_forward - df_ny.Base_price)**2))/len(df_ny)

        # ans.append(sum((df_ny.base_forward - df_ny.base_back)**2) - sum((df_ny.base_forward - df_ny.Base_price)**2))
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lineplot(data = r_table)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Difference in MSE")
    plt.savefig('figures/price-trend.eps')
    plt.show()

    # r = 0.75
    # df_ny = df
    # df_ny['Base_price'] = df_ny['base_back'] + r*(df_ny['base_back'] - df_ny['base_back'].shift(24))


def baseprice_21():
    df = l.load_plot_df([22])
    df = calculate_base_price_day(df, 5)
    # df["Pricechange"] = df.Price.sub(df.Price.shift(24))
    # df["Futureprice"] = df.Price.sub(df.Price.shift(-24))
    df['Pricebin'] = pd.qcut(df['Price'], q = 8)
    # ax = sns.violinplot(data=df, x="Year", y="Price", showfliers = False, inner="quartiles", scale="width")
    # sns.scatterplot(data=df, x="Pricechange", y = "Price")
    # plt.show()
    sns.violinplot(data=df, y='Base_price', x = 'Pricebin')
    # plt.ylim(-10, 120)
    plt.show()

def baseprice_back_forth():


def plot_water_level():
    # years = list(range(11,22))
    years = [12,21]
    df = l.load_plot_df(years)
    df['Waterlevelbin'] = pd.qcut(df['Water_magasine'], q = 8, precision=1)
    # df["Futureprice"] = df.Price.sub(df.Price.shift(-24))
    # df['Pricechangebin'] = pd.qcut(df['Pricechange'], q = 8)
    # sns.scatterplot(df, x = 'Water_magasine', y='Price')
    sns.violinplot(data=df, y='SE2', x = 'Waterlevelbin', hue='Year', split=True, inner='quartiles')
    plt.ylim([-20, 1000])
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylabel('Price SEK/MWh')
    plt.xlabel('Water level [%]')
    plt.tight_layout()
    plt.savefig("figures/waterlevel_price_"+str(years).replace(" ", "")+".eps")
    plt.show()

def plot_water_level_years():
    years = list(range(11,23))
    # years = [12,21]
    df = l.load_plot_df(years)
    # df['Waterlevelbin'] = pd.qcut(df['Water_magasine'], q = 8, precision=1)
    # df["Futureprice"] = df.Price.sub(df.Price.shift(-24))
    # df['Pricechangebin'] = pd.qcut(df['Pricechange'], q = 8)
    # sns.scatterplot(df, x = 'Water_magasine', y='Price')
    sns.violinplot(data=df, y='Water_magasine', x = 'Year', inner='quartiles')
    plt.ylim([0, 100])
    ax.tick_params(axis='x', labelrotation=90)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylabel('Water level [%]')
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig("figures/waterlevel_years_11-22.eps")
    plt.show()

def lineplot_waterlevel():
    # years = list(range(11,23))
    years = [18, 19, 20, 21, 22]
    df = l.load_water_levels(years)
    df["Year"] = df.index.year
    sns.lineplot(data=df, x='Week', y='Level', hue='Year', style ='Year')
    plt.tight_layout()
    plt.ylabel('Water level [%]')
    plt.savefig('figures/water_level_weeks.eps')
    plt.show()


def wind_violinplot():
    # df = plot_day_price([21])
    years = [12, 21]
    # years = list(range(11,22))
    df = l.load_plot_df(years)
    df = df[(df.year == 2012) | (df.year == 2021)]
    df["Wind velocity bins"] = pd.qcut(df['WindSpeed'], q = 8, precision=1)
    # df["Wind velocity bins"] = pd.cut(df['Weather average'], bins = 50, precision=1)
    sns.set(rc={'figure.figsize':(9,4)})
    ax = sns.violinplot(data=df, x = 'Wind velocity bins', y='SE2', hue='year', inner='quartiles', width=1, split=True)
    # ax = sns.boxplot(data=df, x = 'Wind velocity bins', y='Price', fliersize=0)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylim([-30, 1100])
    plt.tight_layout()
    plt.ylabel('Price [SEK/MWh]')
    plt.xlabel('Wind velocity, binned [m/s]')
    plt.savefig("figures/wind_price_"+str(years).replace(" ", "")+".eps")
    plt.show()

def wind_plot_close():
    # df = plot_day_price([21])
    years = [21]
    df, stations = l.load_weather_price(years)
    juktan_loc = (65.2902, 17.2301)
    dist = 300 # in km
    cols = []
    # df = l.load_more_prices(years)
    for index, s in stations.iterrows():
        if distance.distance(juktan_loc, (s.lat, s.long)).km < dist:
            cols = cols + ['W ' + s['Name']]
        else: continue
    df['LocalWindMean'] = df[cols].mean(numeric_only=True, axis=1)
    df["LocalWindbin"] = pd.qcut(df['LocalWindMean'], q = 8)
    df["Windbin"] = pd.qcut(df['Wind average'], q = 8)
    plt.figure(1)
    plt.subplot(211)
    ax1 = sns.violinplot(data=df, x = 'Windbin', y='Price', inner='quartiles', width=1)
    plt.ylim([-10, 100])
    plt.subplot(212)
    ax2 = sns.violinplot(data=df, x = 'LocalWindbin', y='Price', inner='quartiles', width=1)
    plt.ylim([-10, 100])
    plt.show()

def temp_plot():
    # df = plot_day_price([21])
    years = [12, 21]
    df = l.load_plot_df(years, 'TemperaturePast1h')
    df.rename(mapper = {"Weather average": 'temp'}, axis=1, inplace=True)
    # df['temp'] = df.temp.round(3)
    # years = list(range(11,22))
    # df = l.load_more_prices(years)
    # df, _ = l.load_weather_price(years, "TemperaturePast1h")
    # df["Year"] = df.index.year
    df["Temp_bins"] = pd.qcut(df['temp'], q = 8, precision=2)
    df["Temp_bins"] = df["Temp_bins"].apply(lambda x: pd.Interval(left=x.left.round(1), right=x.right.round(1)))
    sns.set(rc={'figure.figsize':(9,4)})
    ax = sns.violinplot(data=df, x = 'Temp_bins', y='SE2', inner='quartiles', width=1, hue="Year", split=True)
    ax.tick_params(axis='x', labelrotation=90)
    plt.ylim([-30, 1000])
    plt.ylabel('Price [SEK/MWh]')
    plt.xlabel('Temperature, binned [°C]')
    plt.tight_layout()
    plt.savefig("figures/temp_price_"+str(years).replace(" ", "")+".eps")
    plt.show()

def regression_plot_temp():
    # sns.set_theme(style="ticks")
    years = [19, 20, 21, 22]
    df = l.load_plot_df(years, 'TemperaturePast1h')
    df.rename(mapper = {"Weather average": 'temp'}, axis=1, inplace=True)
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lmplot(data=df, x = 'temp', y = 'SE2', x_bins=8, hue='year', truncate=True, line_kws={'alpha':0.4}, height=4, aspect=9/4, legend=False, order = 2)
    plt.ylabel('Price [SEK/MWh]')
    # plt.ylim([0, 1000])
    # plt.xlim([1, 7])
    plt.xlabel('Average temperature [°C]')
    plt.legend(loc='upper right')
    
    # sns.residplot(data=df, x = 'wind', y = 'Price', lowess=True, line_kws=dict(color="r"))

    # results = smf.ols('Price ~ np.log(wind)', data=df).fit()
    # fig = sm.graphics.plot_ccpr(results, 'np.log(wind)', )
    plt.savefig('figures/temp_regression_19-22.png')
    plt.show()

def regression_plot_wind():
    # sns.set_theme(style="ticks")
    years = [19, 20, 21, 22]
    df = l.load_plot_df(years)
    df.rename(mapper = {"Weather average": 'wind'}, axis=1, inplace=True)
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lmplot(data=df, x = 'wind', y = 'SE2', x_bins=8, hue='year', truncate=True, line_kws={'alpha':0.4}, height=4, aspect=9/4, legend=False)
    plt.ylabel('Price [SEK/MWh]')
    plt.ylim([0, 1000])
    plt.xlim([1, 7])
    plt.xlabel('Average wind velocity [m/s]')
    plt.legend(loc='upper right')
    
    # sns.residplot(data=df, x = 'wind', y = 'Price', lowess=True, line_kws=dict(color="r"))

    # results = smf.ols('Price ~ np.log(wind)', data=df).fit()
    # fig = sm.graphics.plot_ccpr(results, 'np.log(wind)', )
    plt.savefig('figures/wind_regression_19-22.png')
    plt.show()


def wind_dependency_baseprice():
    years = list(range(19, 23))
    # years = [21]
    day_span = 4
    df = l.load_plot_df(years)
    df = om.calculate_base_price_day(df, day_span, zone)
    df = df.rename({'Base_price': 'base_forward'}, axis=1)
    df = om.calculate_base_price_day(df, -day_span, zone)
    df = df.rename({'Base_price': 'base_back'}, axis=1)

    df['wind_mean_back'] = df['WindSpeed'].rolling(24*day_span).mean()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24*day_span)
    df['wind_mean_forward'] = df['WindSpeed'].rolling(indexer).mean()

    df['wind_diff'] = df['wind_mean_forward'] - df['wind_mean_back']
    df['base_diff'] = df['base_forward'] - df['base_back']

    sns.lmplot(data=df, x = 'wind_diff', y = 'base_diff', x_bins=10, scatter_kws={'alpha':0.4}, hue='Year', order=1, height=5, aspect=1, legend=False)
    # sns.kdeplot(data=df, x = 'wind_diff', y = 'base_diff')
    # sns.residplot(data=df, x = 'wind_diff', y = 'base_diff', lowess=True, scatter_kws={'alpha':0.2}, line_kws={'color':'orange'})
    plt.legend(loc='upper right')
    plt.xlabel('Wind difference')
    plt.ylabel('Baseprice difference')
    # plt.savefig('figures/wind_diff-base_diff_year.png')
    plt.show()


def regression_plot_region():
    # sns.set_theme(style="ticks")
    years = [21]
    df = l.load_plot_df(years)
    df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Weather average", 'Year'], value_name='Price', var_name='Region')
    df.rename(mapper = {"Weather average": 'wind'}, axis=1, inplace=True)
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lmplot(data=df, x = 'wind', y = 'Price', x_bins=10, hue='Region', truncate=True, line_kws={'alpha':0.4}, height=4, aspect=9/4, legend=False)
    plt.ylabel('Price [SEK/MWh]')
    plt.ylim([0, 1250])
    plt.xlim([1, 6])
    plt.xlabel('Average wind velocity [m/s]')
    plt.legend(loc='upper right')
    
    # sns.residplot(data=df, x = 'wind', y = 'Price', lowess=True, line_kws=dict(color="r"))

    # results = smf.ols('Price ~ np.log(wind)', data=df).fit()
    # fig = sm.graphics.plot_ccpr(results, 'np.log(wind)', )
    plt.savefig('figures/wind_regression_21_regions.png')
    plt.show()

def export_price_fit_regions():
    years  = [20]
    df = l.load_plot_df(years)
    df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Import/Export", 'Year'], value_name='Price', var_name='Region')
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lmplot(data=df, x='Import/Export', y='Price', hue='Region', x_bins=12, height=4, aspect=9/4, scatter_kws={'alpha':0.4})
    plt.ylabel('Price [SEK/MWh]')
    plt.xlabel('Import/Export [MWh/h]')
    # plt.ylim([0, 2000])
    # plt.xlim([-7000, 2000])
    plt.show()

def export_price_fit_years():
    years  = [18, 19, 20, 21, 22]
    df = l.load_plot_df(years)
    # df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Import/Export", 'Year'], value_name='Price', var_name='Region')
    
    ax = sns.lmplot(data=df, x='Import/Export', y='SE2', hue='Year', x_bins=12, height=4, aspect=9/4, legend=False, scatter_kws={'alpha':0.4})
    plt.ylabel('Price [SEK/MWh]')
    plt.xlabel('Import/Export [MWh/h]')
    plt.ylim([0, 2000])
    plt.xlim([-7000, 2000])
    plt.legend(loc='upper left')
    plt.savefig("figures/export_price_fit_years_18_22_SE3.png")
    plt.show()

def export_price_violin():
    years  = [19, 20, 21]
    df = l.load_plot_df(years)
    df = df.melt(value_vars=['SE1', 'SE4'], id_vars=["Import/Export", 'Year'], value_name='Price', var_name='Region')
    df['Import/Export'] = df['Import/Export'].div(1000)
    df["Export bins"] = pd.qcut(df['Import/Export'], q = 8, precision=1)

    ax = sns.violinplot(data=df, x='Export bins', y='Price', hue='Region', split=True)
    ax.tick_params(axis='x', labelrotation=90)
    plt.tight_layout()
    plt.ylabel('Price [SEK/MWh]')
    plt.xlabel('Import/Export [GWh/h]')
    plt.show()

def export_violin():
    df = l.load_plot_df(list(range(11,23)))
    # df.rename(mapper={'Price SE1': 'SE1', 'Price SE2': 'SE2', 'Price SE3': 'SE3', 'Price SE4': 'SE4'}, axis=1, inplace=True)
    # df = df.melt(value_vars=['SE1', 'SE4'], id_vars=['Year', 'Import/Export'], value_name='Price', var_name='Region')
    sns.set(rc={'figure.figsize':(9,4)})
    ax = sns.violinplot(data=df, x="Year", y="Import/Export", showfliers = False, inner="quartiles", split=True)
    plt.tight_layout()
    plt.ylabel('Import/Export [MWh/h]')
    plt.xlabel('Year')
    plt.savefig('figures/export_violin_years.eps')
    plt.show()

def temp_wind_fit():
    # sns.set_theme(style="ticks")
    years = [19, 20, 21]
    df_temp = l.load_weather_price(years, 'TemperaturePast1h')
    df_temp.rename(mapper = {"Weather average": 'Temp'}, axis=1, inplace=True)
    df_wind = l.load_plot_df(years, 'WindSpeed')
    df_wind.rename(mapper = {"Weather average": 'Wind'}, axis=1, inplace=True)
    df = pd.merge(left = df_wind, right = df_temp, how='inner', left_index=True, right_index=True)
    # df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Temp", 'Wind'], value_name='Price', var_name='Region')

    sns.lmplot(data=df, x = 'Wind', y = 'Temp', x_bins=20, truncate=True, order = 2, scatter_kws={'alpha':0.4})

    plt.show()

def wind_marginal():
    years  = [19, 20, 21]
    df = l.load_plot_df(years)
    df = df.melt(value_vars=['SE1', 'SE2', 'SE3', 'SE4'], id_vars=["Weather average", 'Year'], value_name='Price', var_name='Region')
    sns.catplot(data=df, x = 'Weather average', y = 'Price', hue = 'Region', col='Year', kind="violin")
    # sns.jointplot(data=df, x = 'Weather average', y = 'SE2')
    plt.show()

def energy_production_overview():
    years = list(range(11,23))
    df = l.load_plot_df(years)
    # df['Wind Production'].ewm(span=24).mean()
    
    cols = ['Nuclear Production', 'Water Production', 'Wind Production', 'Heat Production', 'Gas/Diesel production', 'Solar Production', 'Import/Export', 'Total production']
    df_smooth = df[cols].ewm(span=24*30*2).mean()
    map = {'Nuclear Production': 'Nuclear', 'Water Production': 'Water', 'Wind Production': 'Wind', 'Heat Production': 'Heat', 'Gas/Diesel production': 'Gas/Diesel', 'Solar Production': 'Solar'}
    # exponentially weighted
    df_rename = df_smooth.rename(mapper = map, axis=1)
    df_rename = df_rename.div(1000)
    sns.lineplot(data=df_rename)
    # df_smooth.plot.area(stacked=False)
    plt.ylabel('Production [GWh/h]')
    plt.xlabel('Time')
    # df['Total Consumption'].plot.line()
    plt.tight_layout()
    # plt.savefig('figures/')
    plt.show()

def energy_production_week():
    years = [21]
    df = l.load_plot_df(years)
    cols = ['Nuclear Production', 'Wind Production', 'Heat Production', 'Solar Production', 'Water Production']
    map = {'Nuclear Production': 'Nuclear', 'Water Production': 'Water', 'Wind Production': 'Wind', 'Heat Production': 'Heat', 'Solar Production': 'Solar'}
    df_prod = df[cols].rename(mapper = map, axis=1)
    df_prod = df_prod.div(1000)
    df_prod.loc['2021-01-01':'2021-01-08'].plot.area()
    # sns.lineplot(df['Total Consumption'])
    plt.xlabel('Time')
    plt.ylabel('Electricity production [GWh/h]')
    plt.tight_layout()
    plt.show()


def base_price_plot():
    years = [19, 20, 21, 22]
    df = l.load_plot_df(years)
    df['Price'] = df['SE2']
    df = calculate_base_price_day(df, 5)
    df = df.rename({'Base_price': 'base_forward'}, axis=1)
    df = calculate_base_price_day(df, -5)
    df = df.rename({'Base_price': 'base_back'}, axis=1)
    ax = sns.lmplot(data=df, x='base_back', y='base_forward', hue='Year', scatter_kws={'alpha':1}, x_bins=10)
    plt.xlim([0, 1550])
    plt.ylim([0,2100])
    plt.show()

def two_week():
    # years = [19, 20, 21, 22]
    years = list(range(11,23))
    day_span = 4
    zone = 'SE2'
    df = l.load_plot_df(years)
    df['Price'] = df[zone]
    df = om.calculate_base_price_day(df, day_span, zone)
    df = df.rename({'Base_price': 'Baseprice future'}, axis=1)
    df = om.calculate_base_price_day(df, -day_span, zone)
    df = df.rename({'Base_price': 'Baseprice history'}, axis=1)
    t_start = dt.datetime(year=2022, month=12, day=15)
    days = 14
    t_end = t_start + dt.timedelta(days=days)
    data = df.loc[t_start:t_end, [zone, 'Baseprice history', 'Baseprice future']]
    sns.set(rc={'figure.figsize':(9,3.5)})
    sns.lineplot(data=data)
    # sns.lineplot(data=df_window, y = zone, x = df_window.index, legend='Price', format)
    # sns.lineplot(data=df_window, y = 'base_back', x = df_window.index, legend='Base price backwards')
    # sns.lineplot(data=df_window, y = 'base_forward', x = df_window.index, legend='Base price forwards')
    plt.ylabel('Price [SEK/MWh]')
    plt.xlabel('Time')
    plt.tight_layout()

    plt.savefig('figures/base_price_'+str(days)+'_'+str(t_start.date())+'.eps')
    # plt.legend()
    # plt.legend(['Price', 'Base price backwards', 'Base price forwards'])

    plt.show()

def any_twoweek():
    importlib.reload((om))
    years=[20]
    day_span=5
    zone = 'SE2'
    load_years = list(range(years[0]-1, years[-1]+2))
    frame = l.load_weather_price(load_years, 'WindSpeed')
    df = frame.copy()
    df['Price'] = df[zone]
    diff=dt.timedelta(days=15)
    train_span=dt.timedelta(days=90)
    train_run_delay=train_span + dt.timedelta(day_span)
    # df = om.calculate_base_price_wind_rolling(df, day_span, zone, diff, train_span, train_run_delay, smf.rlm)
    df = om.calculate_base_price_trend_randomwalk(df, day_span, zone)
    t = dt.datetime(year=2020, month=1, day=1)
    days = 365
    t_e = t + dt.timedelta(days=days)
    data = df.loc[t:t_e, ['Price', 'base_forward', 'base_back', 'Base_price']]
    data.columns = ['Price', 'Future base price', 'History base price', 'History trend base price']
    # data.columns = ['Price', 'Future base price', 'History base price', 'Predicted base price']
    sns.set(rc={'figure.figsize':(5,6)})
    ax = sns.lineplot(data=data)
    plt.xticks([t, t_e])
    plt.xlabel('Time')
    plt.ylabel('Price [SEK/MWh]')
    plt.tight_layout()
    # plt.savefig('figures/two-week-trend.eps')
    plt.show()
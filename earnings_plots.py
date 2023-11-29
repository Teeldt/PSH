# Plots for the earnings algorithm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import statsmodels.formula.api as smf
from scipy import stats
import cProfile
import pstats
from pstats import SortKey
import operations_model as om
import loader as l
import importlib
sns.set_theme()
importlib.reload(om)
importlib.reload(l)

def create_res_table(full_years, list_dfs):
    '''To get information for dataframes and format it in a plottable way. When comparing different years.'''
    # full_years = [i + 2000 for i in years]
    table = pd.DataFrame(index=full_years)
    table['Year'] = table.index
    for df in list_dfs:
        for y in full_years:
            table.loc[y, df.name ] = df[df.index.year==y].Income[-1]/(10**6)
            table.loc[y, df.name + ', After grid tariff'] = df[df.index.year==y].IncomeAfterTax[-1]/(10**6)
            table.loc[y, df.name + ', Generation'] = df[df.index.year==y].Generation[-1]
    return table

def create_res_table_variable(full_years, dict_dfs, divides = 1, extra_cols=[]):
    '''To get information for dataframes and format it in a plottable way. When comparing different parameter values (as keys in the dict), dividing into time periods.'''
    table = pd.DataFrame(columns=['Variable', 'Income', 'After grid tariff', 'Generation', 'Year', 'Starttime', 'Midtime', 'Length'] + extra_cols, index=range(0, len(full_years)*len(dict_dfs)*(divides+1)))
    # periods = (365 // divides)*24
    i = 0

    for key, df in dict_dfs.items():
        for y in full_years:
            y_df = df[df.index.year==y]
            # power_tariff = y_df.Tax[0]/divides
            # y_df.loc[:, 'Tax'] = y_df['Tax'] - y_df.iloc[0].Tax
            # split into chunks
            periods = int(((y_df.last_valid_index().dayofyear *24)/ divides))
            parts = [y_df.iloc[j:j+periods] for j in range(0,len(y_df),periods)]
            for p in parts:
                # print(table)
                # print(p.first_valid_index())
                table['Variable'].iloc[i] = key
                table['Income'].iloc[i] = p.Income[-1]/(10**6)
                # tax only working per year
                if divides == 1:
                    table['After grid tariff'].iloc[i] = p.IncomeAfterTax[-1]/(10**6)
                table['Generation'].iloc[i] = sum(p.Generation)
                table['Year'].iloc[i] = y
                table['Starttime'].iloc[i] = p.first_valid_index()
                table['Length'].iloc[i] = p.last_valid_index() - p.first_valid_index()
                table['Midtime'].iloc[i] = table['Starttime'].iloc[i] + table['Length'].iloc[i]/2
                for col in extra_cols:
                    table[col].iloc[i] = sum(p[col])
                # table['Endtime'].iloc[i] = p.last_valid_index()
                # table['Length'].iloc[i] = p.last_valid_index() - p.first_valid_index()
                
                i = i+1
    return table.dropna()

# for p in range(0, divides):
#             ans_df.income.loc[i] = parts[p+1].Income[0] - parts[p].Income[0]
#             ans_df.generation.loc[i] = parts[p+1].Total_generation[0] - parts[p].Total_generation[0]
#             ans_df.mean_price.loc[i] = parts[p].Price.mean()
#             ans_df.deviation.loc[i] = parts[p].Price.std()
#             ans_df.waterchange.loc[i] = (parts[p+1].Storage[0] - parts[p].Storage[0])
#             ans_df.earnings.loc[i] = parts[p+1].Income[0] - parts[p].Income[0] + parts[p].Price.mean()*(parts[p+1].Storage[0] - parts[p].Storage[0])
#             ans_df.start_time.loc[i] = parts[p].first_valid_index()
#             ans_df.end_time.loc[i] = parts[p].last_valid_index()
#             i += 1

def test_days():
    path = 'data/day_table.pkl'
    try:
        day_table = pd.read_pickle(path)
    except:
        years = [13, 16, 19, 22]
        full_years = [i + 2000 for i in years]
        days_ahead = range(-7, 10)
        dfs = {}
        for d in days_ahead:
            dfs[d] = om.calculate_earnings(years, om.calculate_base_price_day, d)
        day_table = create_res_table_variable(full_years, dfs)
        day_table['Norm income'] = pd.Series()
        for y in full_years:
            day_table.loc[day_table['Year'] == y, 'Norm income'] = day_table.loc[day_table['Year'] == y, 'Income'] / np.mean(day_table.loc[day_table['Year'] == y, 'Income'])

        day_table.to_pickle(path)
    sns.set(rc={'figure.figsize':(9,4)})
    sns.lineplot(data=day_table, x = 'var', y = 'Norm income', hue='Year', markers=True)
    plt.xlabel('Days ahead')
    plt.ylabel('Income [normalized]')
    plt.tight_layout()
    plt.savefig('figures/earnings_day_ahead.eps')
    plt.show()

def compare_years_simple():
    years = list(range(11,23))
    full_years = [i + 2000 for i in years]
    day_span = 4

    df_forward = om.calculate_earnings(years, om.calculate_base_price_day, day_span)
    df_forward.name = 'Future baseprice'
    df_back = om.calculate_earnings(years, om.calculate_base_price_day, -day_span)
    df_back.name = 'History baseprice'
    # df_wind = om.calculate_earnings(years, om.calculate_base_price_wind, day_span)
    # df_wind.name = 'bp_w'
    
    
    table = create_res_table(full_years, [df_forward, df_back])
    # table.to_pickle('data/compare_years_335,335,25000,1,1.pkl')
    data = table.loc[:, ['History baseprice', 'Future baseprice', 'History baseprice, After grid tariff', 'Future baseprice, After grid tariff']]
    # data = data.melt(value_vars=['History baseprice, Income', 'Future baseprice, Income'], id_vars=['Weather average', 'Year'], value_name='Income', var_name='Region')
    sns.lineplot(data=data, markers=True)
    # legend=['Base price backward income', 'Base price forward income', 'Base price backward after grid tarrif', 'Base price forward after grid tarrif']
    plt.ylabel("Income [MSEK]")
    plt.xlabel('Year')
    plt.tight_layout()
    # plt.ylabel("Total generation [GWh]")
    # plt.savefig("figures/earnings_per_year.eps")
    plt.show()
    # plt.show()

# # to set optional values
# def calculate_base_price_trend_wrapper(df, day_span, zone):
#     importlib.reload(om)
#     diff = dt.timedelta(days=15)            # how large period each training set is used for
#     train_span = dt.timedelta(days=90)     # how large each training set is. If diff+train_span+day_span is larger than one year, errors within the studied period.
#     train_run_delay = train_span + dt.timedelta(day_span)
#     return om.calculate_base_price_trend(df, day_span, zone, diff, train_span, train_run_delay)

# def calculate_base_price_trend_incl_wrapper(df, day_span, zone):
#     importlib.reload(om)
#     diff = dt.timedelta(days=15)            # how large period each training set is used for
#     train_span = dt.timedelta(days=90)     # how large each training set is. If diff+train_span+day_span is larger than one year, errors within the studied period.
#     train_run_delay = train_span/2
#     return om.calculate_base_price_trend(df, day_span, zone, diff, train_span, train_run_delay)

# def calculate_base_price_wind_wrapper(df, day_span, zone):
#     importlib.reload(om)
#     diff = dt.timedelta(days=7)            # how large period each training set is used for
#     train_span = dt.timedelta(days=90)     # how large each training set is. If diff+train_span+day_span is larger than one year, errors within the studied period.
#     train_run_delay = train_span + dt.timedelta(day_span)
#     return om.calculate_base_price_wind_rolling(df, day_span, zone, diff, train_span, train_run_delay, smf.ols)

# def calculate_base_price_wind_robust_wrapper(df, day_span, zone):
#     importlib.reload(om)
#     diff = dt.timedelta(days=30)            # how large period each training set is used for
#     train_span = dt.timedelta(days=90)     # how large each training set is. If diff+train_span+day_span is larger than one year, errors within the studied period.
#     train_run_delay = train_span + dt.timedelta(day_span)
#     return om.calculate_base_price_wind_rolling(df, day_span, zone, diff, train_span, train_run_delay, smf.rlm)


def compare_methods():
    years = list(range(18,23))
    full_years = [i + 2000 for i in years]
    day_span = 4

    df_forward = om.calculate_earnings(years, om.calculate_base_price_day, day_span)
    df_forward.name = 'Future baseprice'
    df_back = om.calculate_earnings(years, om.calculate_base_price_day, -day_span)
    df_back.name = 'History baseprice'
    # df_trend_inc = om.calculate_earnings(years, calculate_base_price_trend_incl_wrapper, day_span)
    # df_trend_inc.name = 'Future trend baseprice'
    df_trend = om.calculate_earnings(years, om.calculate_base_price_trend_randomwalk, day_span)
    df_trend.name = 'History trend baseprice'
    # df_wind = om.calculate_earnings(years, calculate_base_price_wind_wrapper, day_span)
    # df_wind.name = 'Prediction wind baseprice'
    # df_wind_rob = om.calculate_earnings(years, calculate_base_price_wind_robust_wrapper, day_span)
    # df_wind_rob.name = 'Prediction wind baseprice, robust'

    # table = create_res_table(full_years, [df_forward, df_back, df_trend, df_trend_inc, df_wind, df_wind_rob])
    table = create_res_table(full_years, [df_forward, df_back, df_trend])
    # table.to_pickle('data/compare_years_335,335,25000,1,1.pkl')
    # data = table.loc[:, ['History baseprice', 'Future baseprice', 'History trend baseprice', 'Future trend baseprice', 'Prediction wind baseprice']]
    # data = table.loc[:, ['History baseprice', 'Future baseprice', 'Prediction wind baseprice', 'Prediction wind baseprice, robust']]
    data = table.loc[:, ['History baseprice', 'Future baseprice', 'History trend baseprice']]
    sns.set(rc={'figure.figsize':(5,6)})
    sns.lineplot(data=data, markers=True)
    plt.ylabel("Income [MSEK]")
    plt.xlabel('Year')
    plt.ylim([0, 400])
    plt.tight_layout()
    # plt.savefig("figures/compare_methods_trend_earnings_r075.eps")
    plt.show()


def mean_or_median():
    dfs_mm = ['forward_mean', 'back_mean', 'forward_median', 'back_median']
    try:
        names = ['forward_mean', 'back_mean', 'forward_median', 'back_median']
        dfs_mm = []
        for n in names:
            dfs_mm.append(pd.read_pickle('data/pickles/dfs_mm_'+n+'.pkl'))
    except:
        years = list(range(11,23))
        days_ahead = 4
        full_years = [i + 2000 for i in years]
        # mean_dfs = []
        df_forward_mean = om.calculate_earnings(years, om.calculate_base_price_day, days_ahead)
        df_forward_mean.name = 'Future mean'
        df_back_mean = om.calculate_earnings(years, om.calculate_base_price_day, days_ahead)
        df_back_mean.name = 'History mean'
        df_forward_median = om.calculate_earnings(years, om.calculate_base_price_day_median)
        df_forward_median.name = 'Future median'
        df_back_median = om.calculate_earnings(years, om.calculate_base_price_day_median)
        df_back_median.name = 'History median'

        dfs_mm = [df_forward_mean,df_back_mean, df_forward_median, df_back_median]
        for df in dfs_mm:
            df.to_pickle('data/pickles/dfs_mm_'+df.name+'.pkl')

    table_mm = create_res_table(full_years, dfs_mm)
    t_mm = table_mm.loc[:, ['Future mean, Income', 'History mean, Income', 'Future median, Income', 'History median, Income']]
    sns.lineplot(data=t_mm)
    plt.ylabel('Income [MSEK]')
    plt.xlabel('Year')
    plt.tight_layout
    plt.savefig('figures/mean_or_median.eps')
    plt.show()

def plot_function():
    res_df = om.calculate_earnings(22, 1)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time')
    ax1.plot(res_df.Storage, color='blue')
    ax1.set_ylabel("Storage [MWh]")
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(res_df.Buy_price, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel("Price [SEK/MWh]")
    # pl = res_df.plot(y="Storage")
    # plt.boxplot(x = res_df["Wind average"], y = res_df["Price"])
    fig.tight_layout()

    plt.show()


def test_rate():
    earnings = []
    rates = np.arange(0.85, 1.15, 0.02)
    for r in rates:
        res_df = om.calculate_earnings(20, r)
        earnings.append(res_df.Income[-1])
    plt.plot(rates, earnings)
    plt.show()

# def get_results(df):
#     for year in 

def test_days(years=[13, 16, 19, 22]):
    dfs = {}
    earnings = []
    generation = []
    days_ahead = range(-7, 10)
    for d in days_ahead:
        dfs[d] = om.calculate_earnings(years, om.calculate_base_price_day, d)
    table = pd.DataFrame()
        # e.append(res_df.Income[-1])
        # generation.append(res_df.Total_generation[-1])

    for y in years:
        e = []
        for d in days_ahead:
            res_df = om.calculate_earnings(y, om.calculate_base_price_day, d)
            e.append(res_df.Income[-1])
            generation.append(res_df.Total_generation[-1])
        earnings.append(e)
        # plt.plot(days_ahead, e/np.mean(e))
    for e in earnings:
        plt.plot(days_ahead, e/np.mean(e))
    plt.legend(years)
    plt.xlabel("Days ahead")
    plt.ylabel("Normalised earnings")
    # plt.plot(days_ahead, generation)
    plt.savefig("figures/earnings_day_ahead.eps")
    plt.close()
    # plt.show()

def corrolate_earnings():
    years = list(range(17, 23))
    divides=12 # per year
    days_ahead = 5
    # divides = 12 # how many parts to split each year into
    periods = (365 // divides)*24
    elements = (len(years))*divides
    ans_df = pd.DataFrame(np.nan, index=range(0, elements),
                          columns=['income', 'generation', 'mean_price', 'deviation', 'waterchange', 'earnings', 'start_time', 'end_time'])
    pd.options.mode.chained_assignment = None

    # for y in years:
    df_for = om.calculate_earnings(years, om.calculate_base_price_day, days_ahead)
    df_back = om.calculate_earnings(years, om.calculate_base_price_day, -days_ahead)
    df_wind = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, days_ahead)

    df = df_wind # change to the one you want to consider
    
    ans_df = pd.DataFrame(np.nan, index=range(0, elements), columns=['income', 'generation', 'mean_price', 'deviation', 'waterchange', 'earnings', 'start_time', 'end_time'])
    pd.options.mode.chained_assignment = None
    i = 0
    for y in years:
        res_df = df[df['year'] == y+2000]
        parts = [res_df.iloc[j:j+periods] for j in range(0,len(res_df),periods)]
        # split into chunks
        for p in range(0, divides):
            ans_df.income.loc[i] = parts[p+1].Income[0] - parts[p].Income[0]
            ans_df.generation.loc[i] = parts[p+1].Total_generation[0] - parts[p].Total_generation[0]
            ans_df.mean_price.loc[i] = parts[p].Price.mean()
            ans_df.deviation.loc[i] = parts[p].Price.std()
            ans_df.waterchange.loc[i] = (parts[p+1].Storage[0] - parts[p].Storage[0])
            ans_df.earnings.loc[i] = parts[p+1].Income[0] - parts[p].Income[0] + parts[p].Price.mean()*(parts[p+1].Storage[0] - parts[p].Storage[0])
            ans_df.start_time.loc[i] = parts[p].first_valid_index()
            ans_df.end_time.loc[i] = parts[p].last_valid_index()
            i += 1

    pd.options.mode.chained_assignment = "warn"

    ans_df['Year'] = pd.to_datetime(ans_df.start_time).dt.year
    plot_df = ans_df.dropna()
    plot_df['Year'] = plot_df['Year'].astype(int)
    ans_df.dropna(inplace=True)
    plot_df = plot_df[ans_df['Year'] != 2017]
    plot_df = plot_df[plot_df['Year'] != 2018]
    plot_df = plot_df[plot_df['Year'] != 2023]
    
    sns.lmplot(data=plot_df, x='deviation', y='earnings', hue='Year', height=5, aspect=1, legend_out=False)
    plt.ylabel('Income [SEK]')
    plt.xlabel('Standard deviation [SEK/MWh]')
    plt.legend(loc='upper left')
    # plt.xscale('log')
    # plt.yscale('log')
    plt.savefig('figures/std_wind_2019-20222.png')
    plt.show()

    # plt.plot(earnings)
    
    # plt.scatter(ans_df.deviation, earnings)
    # plt.scatter(ans_df.deviation, ans_df.generation)
    # res = stats.linregress(ans_df.deviation, ans_df.income)
    # plt.plot(ans_df.deviation, res.intercept + res.slope*ans_df.deviation)
    # plt.plot(days_ahead, generation)
    # plt.show()
    return ans_df

        
def buy_sell_curves_plot(rate):
    x = np.linspace(0, 1, 10)

    plt.plot(x, om.sell_curve(x, rate), color='red')
    plt.plot(x, om.buy_curve(x, rate), color='blue')

    plt.show()

def get_time_profile():
    cProfile.run('om.calculate_earnings()', 'restats')
    p = pstats.Stats('restats')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(0.1)
    return p

def compare_generation():

    maxgens = [25, 50, 100, 200, 400]
    years = list(range(18, 23))
    full_years = full_years = [i + 2000 for i in years]

    dfs_gen = {}
    for m in maxgens:
        dfs_gen[m] = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, m, m*255/355)

    # gen_table = create_res_table_variable(full_years, dfs_gen)
    # sns.lineplot(data=gen_table, x = 'Year', y='Income', hue='var')
    # plt.show()

    # barplot
    gen_table = create_res_table_variable(full_years, dfs_gen, 1)
    gen_table.rename({'Variable': 'Maximal generation [MW]'}, axis=1, inplace=True)


    sns.set(rc={'figure.figsize':(9,4)})
    sns.barplot(data=gen_table, x = 'Year', y='Income', hue='Maximal generation [MW]')
    plt.ylabel('Income [MSEK]')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('figures/compare_generation_1.eps')
    plt.show()
    # include tax
    fig, ax = plt.subplots()
    ax = sns.barplot(data=gen_table, x = 'Year', y='Income', hue='Maximal generation [MW]', palette='pastel')
    ax = sns.barplot(data=gen_table, x = 'Year', y='After grid tariff', hue='Maximal generation [MW]')
    h, l = ax.get_legend_handles_labels()
    plt.legend(handles=h, labels=maxgens)
    plt.ylabel('Income [MSEK]')
    plt.tight_layout()
    plt.savefig('figures/compare_generation_1_tax.eps')
    plt.show()

    return

def compare_pump():

    maxpump = [0.7, 0.8, 0.9,1.0,1.1, 1.2]
    years = list(range(18, 23))
    full_years = full_years = [i + 2000 for i in years]

    dfs_pump = {}
    maxgen = 355
    for m in maxpump:
        dfs_pump[m] = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, maxgen, int(m*maxgen))

    # barplot
    pump_table = create_res_table_variable(full_years, dfs_pump, 1)
    pump_table.rename({'Variable': 'Pump'}, axis=1, inplace=True)
    sns.set(rc={'figure.figsize':(9,4)})

    sns.barplot(data=pump_table, x = 'Year', y='Income', hue='Pump')
    plt.ylabel('Income [MSEK]')
    plt.xlabel('Time')
    plt.tight_layout()
    # plt.savefig('figures/compare_pump_1.eps')
    plt.show()

    # include tax
    fig, ax = plt.subplots()
    ax = sns.barplot(data=pump_table, x = 'Year', y='Income', hue='Pump', palette='pastel')
    ax = sns.barplot(data=pump_table, x = 'Year', y='After grid tariff', hue='Pump')
    h, l = ax.get_legend_handles_labels()
    plt.legend(handles=h, labels=maxpump)
    plt.ylabel('Income [MSEK]')
    plt.tight_layout()
    plt.savefig('figures/compare_pump_1_tax.eps')
    plt.show()

    return

def compare_efficiency():

    eff = [0.7,0.8, 0.9, 1.0]
    years = list(range(18, 23))
    full_years = full_years = [i + 2000 for i in years]

    dfs_eff = {}
    for e in eff:
        dfs_eff[e] = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, 335, 255, 25000, np.sqrt(e), np.sqrt(e))

    # gen_table = create_res_table_variable(full_years, dfs_gen)
    # sns.lineplot(data=gen_table, x = 'Year', y='Income', hue='var')
    # plt.show()

    # barplot
    eff_table = create_res_table_variable(full_years, dfs_eff, 1)
    eff_table.rename({'Variable': 'Efficiency'}, axis=1, inplace=True)
    sns.set(rc={'figure.figsize':(9,4)})
    sns.barplot(data=eff_table, x = 'Year', y='Income', hue='Efficiency')
    plt.ylabel('Income [MSEK]')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('figures/compare_eff_1.eps')
    plt.show()
    return

def compare_region():

    regions = ['SE1', 'SE2', 'SE3', 'SE4']
    years = list(range(18, 23))
    full_years = full_years = [i + 2000 for i in years]

    dfs_region = {}
    for r in regions:
        dfs_region[r] = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, 335, 255, 25000, 0.9, 0.9, r)

    # gen_table = create_res_table_variable(full_years, dfs_gen)
    # sns.lineplot(data=gen_table, x = 'Year', y='Income', hue='var')
    # plt.show()

    # barplot
    region_table = create_res_table_variable(full_years, dfs_region, 1)
    region_table.rename({'Variable': 'Region'}, axis=1, inplace=True)
    sns.set(rc={'figure.figsize':(9,4)})
    sns.barplot(data=region_table, x = 'Year', y='Income', hue='Region')
    plt.ylabel('Income [MSEK]')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('figures/compare_region_1.eps')
    plt.show()
    return

def compare_storage():

    maxstorage = [1000, 2000, 4000, 8000, 16000, 32000]
    years = list(range(18, 23))
    full_years = [i + 2000 for i in years]

    res_dfs_storage = {}
    for m in maxstorage:
        res_dfs_storage[m] = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, 335, 255, m)
    
    # divides=6
    # storage_table = create_res_table_variable(full_years, res_dfs_storage, divides)
    # storage_table.rename({'Variable': 'Storage'}, axis=1, inplace=True)
    # sns.lineplot(data=storage_table, x = 'Midtime', y='Income', hue='Storage', style='Storage')
    # # plt.savefig('figures/compare_storage_'+str(divides)+'.eps')
    # plt.show()

    # barplot
    storage_table = create_res_table_variable(full_years, res_dfs_storage, 1)
    storage_table.rename({'Variable': 'Storage'}, axis=1, inplace=True)
    sns.set(rc={'figure.figsize':(9,4)})
    sns.barplot(data=storage_table, x = 'Year', y='Income', hue='Storage')
    plt.ylabel('Income [MSEK]')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig('figures/compare_storage_1.eps')
    plt.show()
    return

def three_reservoir():
    years = list(range(18, 23))
    full_years = [i + 2000 for i in years]
    high = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, 335, 255, 25000, 1.0, 1.0, 'SE2')
    low = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, 265, 255, 25000, 0.9, 0.9, 'SE2')
    res_dfs = {"Storjuktan-Blaiksjön-Storuman": high, 'Storjuktan-Blaiksjön-Storjuktan': low}
    
    res_table = create_res_table_variable(full_years, res_dfs)

    # table = pd.DataFrame(columns=['Income', 'After grid tariff', 'Generation', 'Year'], index=range(0, len(full_years)*len(dict_dfs)*(divides+1)))

    high_total_generation = 315*1000 # MWh

    income_highfraction = res_table[res_table.Variable == 'Storjuktan-Blaiksjön-Storuman'].Income*high_total_generation/res_table[res_table.Variable == 'Storjuktan-Blaiksjön-Storuman'].Generation
    income_lowfraction = res_table[res_table.Variable == 'Storjuktan-Blaiksjön-Storjuktan'].Income.values*(1-high_total_generation/res_table[res_table.Variable == '"Storjuktan-Blaiksjön-Storuman"'].Generation)

    df2 = pd.DataFrame({'Variable': ['Result']*5,
                        'Income': (income_highfraction + income_lowfraction).values,
                        'Year': full_years})
                        
    
    full_table = pd.concat([res_table, df2], ignore_index=True)
    full_table.rename({'Variable': 'Operation mode'}, axis=1, inplace=True)

    sns.set(rc={'figure.figsize':(9,4)})
    sns.catplot(data=full_table, kind='bar', x='Year', y='Income', hue='Operation mode', legend_out=False)
    plt.ylabel('Income [MSEK]')
    plt.xlabel('Year')
    plt.savefig('figures/three-reservoir.eps')
    plt.show()
    

def frequency_regulation():
    years = list(range(18, 24))
    full_years = [i + 2000 for i in years]
    operations = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, 335, 255, 25000, 0.9, 0.9, 'SE2')

    fcr = l.load_fcr_prices()
    df = pd.merge(left=operations, right=fcr, right_index=True, left_index=True)

    df['FCR-D Up'] = (df.Generation + df.Pumping)/0.9*0.1*df['FCR-D_up']/(10**6)
    df['FCR-D Down'] = (df.Generation + df.Pumping)/0.9*0.1*df['FCR-D_down']/(10**6)
    res_dfs = {'Operations': df}

    res_table = create_res_table_variable(full_years, res_dfs, 1, ['FCR-D Up', 'FCR-D Down'])
    data = res_table.loc[:, ['FCR-D Up', 'FCR-D Down', 'Income', 'After grid tariff']]
    data['Sum'] = data['After grid tariff'] + data['FCR-D Up'] + data['FCR-D Down']

    sns.set(rc={'figure.figsize':(5,6)})
    data.index=full_years
    sns.lineplot(data=data, markers=True)
    plt.ylabel("Income [MSEK]")
    plt.xlabel('Year')
    plt.ylim([-50, 450])
    plt.tight_layout()
    plt.savefig('figures/FCR-potential.eps')
    plt.show()

def frequency_regulation_compare():
    years = list(range(18, 24))
    full_years = [i + 2000 for i in years]
    operations = om.calculate_earnings(years, om.calculate_base_price_wind_rolling, 4, 335, 255, 25000, 0.9, 0.9, 'SE2')
    fcr = l.load_fcr_prices()
    df = pd.merge(left=operations, right=fcr, right_index=True, left_index=True)

    fcr_capacity = [0, 0.05, 0.10, 0.15, 0.20]

    res_dfs = {}
    for c in fcr_capacity:
        res_dfs[c] = df.copy()
        res_dfs[c]['FCR-D Up'] = (df.Generation + df.Pumping)*c*df['FCR-D_up']/(10**6)
        res_dfs[c]['FCR-D Down'] = (df.Generation + df.Pumping)/0.9*c*df['FCR-D_down']/(10**6)
    
    table = create_res_table_variable(full_years, res_dfs, 1, ['FCR-D Up', 'FCR-D Down'])
    table.rename({'Variable': 'Part of capacity'}, axis=1, inplace=True)
    sns.lineplot(data=table, x = 'Year', y='FCR-D Up', hue='Part of capacity')
    sns.lineplot(data=table, x = 'Year', y='FCR-D Down', hue='Part of capacity', linestyle='--', legend=False)
    plt.ylabel("FCR-D Income [MSEK]")
    plt.xlabel('Year')
    plt.tight_layout()
    plt.savefig('figures/FCR-variation.eps')
    plt.show()

# def compare_storage():
#     maxstorage = [5000, 10000, 15000, 20000, 25000, 30000]
#     res_dfs = {}
#     regs = {}
#     for m in maxstorage:
#         res_dfs[m] = corrolate_earnings(list(range(11,23)), 1, 335, m)
#         plt.scatter(res_dfs[m].deviation, res_dfs[m].earnings, label=m)
#         # plt.scatter(ans_df.deviation, ans_df.generation)
#         reg = stats.linregress(res_dfs[m].deviation, res_dfs[m].earnings)
#         regs[m] = reg
#         plt.plot(res_dfs[m].deviation, reg.intercept + reg.slope*res_dfs[m].deviation)
#     plt.show()

#     slopes = []
#     intercepts = []
#     for m in maxstorage:
#         slopes.append(regs[m].slope)
#         intercepts.append(regs[m].intercept)
#     plt.show()
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import loader as l
import importlib
importlib.reload(l)


def calculate_base_price(in_df, day_span, zone):
    '''Calculate a price to use as buy-sell price reference day-by-day,
    by calculating the average a specified time in advance/back in time
    in_data: a pandas dataframe with hourly prices
    day_span: which timespan to consider, in days. If negative, base on earlier data. If positive, base on future data. If 0, only data of today'''
    if day_span < 0:
        df['Base_price'] = df[zone].rolling(-24*day_span).mean()
    elif day_span >= 0:
        day_span = day_span +1
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=-24*day_span)
        df['Base_price'] = df[zone].rolling(indexer).mean()
    return df

def calculate_base_price_day_median(df, day_span, zone):
    if day_span == 0:
        # rolling takes rolling values with window of a day, but only once a day (step). Using mean of the window. Then fill the missing values (any value not at the beginning of the day)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
        df['Base_price'] = df[zone].rolling(indexer, step=24).median()
        df['Base_price'].fillna(method='ffill', inplace=True)
    elif day_span < 0:
        sh = 24
        df['Base_price'] = df[zone].shift(-sh).rolling(-24*(day_span+1), step=24).median()
        df['Base_price'].fillna(method='ffill', inplace=True)
    elif day_span > 0:
        day_span = day_span +1
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24*(day_span+1))
        df['Base_price'] = df[zone].rolling(indexer, step=24).median()
        df['Base_price'].fillna(method='ffill', inplace=True)
    return df

def calculate_base_price_day(df, day_span, zone):
    '''Calculate a price to use as buy-sell price reference day-by-day,
    by calculating the average a specified time in advance/back in time
    in_data: a pandas dataframe with hourly prices
    day_span: which timespan to consider, in days. If negative, base on earlier data. If positive, base on future data. If 0, only data of today'''
    if day_span == 0:
        # rolling takes rolling values with window of a day, but only once a day (step). Using mean of the window. Then fill the missing values (any value not at the beginning of the day)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
        df['Base_price'] = df[zone].rolling(indexer, step=24).mean()
        df['Base_price'].fillna(method='ffill', inplace=True)
    elif day_span < 0:
        sh = 24
        df['Base_price'] = df[zone].shift(-sh).rolling(-24*(day_span-1), step=24).mean()
        df['Base_price'].fillna(method='ffill', inplace=True)
    elif day_span > 0:
        day_span = day_span +1
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24*(day_span+1))
        df['Base_price'] = df[zone].rolling(indexer, step=24).mean()
        df['Base_price'].fillna(method='ffill', inplace=True)
    return df

def calculate_base_price_trend_randomwalk(df, day_span, zone):
    """Using a random walk method to calculate a base price, in respect to price trends"""
    day_span_back = day_span
    # day_span_forward = 4
    df = calculate_base_price_day(df, day_span, zone)
    df = df.rename({'Base_price': 'base_forward'}, axis=1)
    df = calculate_base_price_day(df, -day_span, zone)
    df = df.rename({'Base_price': 'base_back'}, axis=1)
    r = 0.75
    df['Base_price'] = df['base_back'] + r*(df['base_back'] - df['base_back'].shift(24))
    # try with second degree equation
    # df['Base_price'] = df['base_back'] + r*(df['base_back'] - 1/2*df['base_back'].shift(24) - 1/2*df['base_back'].shift(48))
    # (sum((df.base_forward - df.base_back)**2) - sum((df.base_forward - df.Base_price)**2))/len(df)
    return df

def calculate_base_price_wind_rolling(df, day_span=4, zone='SE1', diff=dt.timedelta(days=7), train_span=dt.timedelta(days=90), train_run_delay= dt.timedelta(days=4+90), regress=smf.rlm):
    """Using a regression method to calculate a base price, taking future weather into account"""
    # diff = dt.timedelta(days=15)            # how large period each training set is used for
    # train_span = dt.timedelta(days=180)     # how large each training set is. If diff+train_span+day_span is larger than one year, errors within the studied period.

    t_start = df.first_valid_index()
    t_end = df['WindSpeed'].last_valid_index()
    dur = (t_end - t_start) // diff

    df = calculate_base_price_day(df, day_span, zone)
    df = df.rename({'Base_price': 'base_forward'}, axis=1)
    df = calculate_base_price_day(df, -day_span, zone)
    df = df.rename({'Base_price': 'base_back'}, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    df['wind_mean_back'] = df['WindSpeed'].rolling(24*day_span).mean()
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=24*day_span)
    df['wind_mean_forward'] = df['WindSpeed'].rolling(indexer).mean()

    df['wind_diff'] = df['wind_mean_forward'] - df['wind_mean_back']

    df['base_diff'] = df['base_forward'] - df['base_back']
    # baseprice = df.Series(index)
    for p in range(0, dur):
        t_begin_train = t_start + p*diff                                    # set train starting time, changing diff every iteration
        t_stop_train = t_begin_train + train_span - dt.timedelta(hours=1)     # set train end time, train_span later

        train_set = df.loc[t_begin_train:t_stop_train, [zone,  'base_forward', 'base_back', 'base_diff', 'wind_diff']]
        res = regress('base_diff ~ wind_diff', data = train_set).fit()
        ps = res.params

        # run time is after the train time, and also after day_span in order to not run on train data
        t_begin_run = t_begin_train + train_run_delay
        # stops diff later, then new data is used
        t_stop_run = t_begin_run + diff
        # t_stop_run = t_start + (p+2)*diff + dt.timedelta(days=day_span, hours=-1)

        df.loc[t_begin_run:t_stop_run, 'Base_price'] = df.loc[t_begin_run:t_stop_run, 'base_back'] + ps.Intercept + ps['wind_diff']*df.loc[t_begin_run:t_stop_run, 'wind_diff']
    return df

def calculate_earnings(years=[22], method=calculate_base_price_day, day_span=4, maxgen=335, maxpump=255, maxstorage=25000, pump_eff = 0.9, gen_eff = 0.9, zone = 'SE2'):
    """ The main function for running the model of a Pumped storage hydroelectricity power station. Returns a dataframe with details about the simulated operations.
        years: list of years to consider
        method: base price method, as a function name
        day_span: days ahead to consider when setting base price. Negative value for earlier days.
        maxgen: maximum generation when in generation mode, in MW
        maxpump: maximum pumping power when in pumping mode, in MW
        maxstorage: available storage capacity in upper reservoir, in MWh
        pump_eff: efficiency of the pump
        gen_eff: efficiency of the generator
        zone: price region considered"""
    # Import prices
    load_years = list(range(years[0]-1, years[-1]+2))
    frame = l.load_weather_price(load_years, 'WindSpeed')
    df = frame.copy()
    df['Price'] = df[zone]
    df = method(df, day_span, zone) # baseprice calculations
    tax_df = l.load_taxes()

    minstorage = maxgen
    # rate = 1.0
    water_level_start = 0.5 # start each year like this

    df['Storage'] = pd.Series(dtype='double')
    df['Total_generation'] = pd.Series(dtype='double')
    # df.loc[df.index[0], 'Total_generation'] = 0
    df['Buy_price'] = pump_eff*df['Base_price']
    df['Sell_price'] = gen_eff*df['Base_price']
    df['Generation'] = pd.Series(dtype='double')
    df['Pumping'] = pd.Series(dtype='double')
    df['Income'] = pd.Series(dtype='double')   
    df['hourIncome'] = pd.Series(dtype='double')
    df['Tax']  = pd.Series(dtype='double')
    
    for y in years:
        start_index = df[df.year == (y+2000)].first_valid_index()
        df.loc[start_index, 'Total_generation'] = 0
        df.loc[start_index, 'Income'] = 0
        df.loc[start_index, 'Storage'] = maxstorage*water_level_start
        df.loc[start_index, 'Tax'] = tax_df.loc[y, 'pump_tax']*maxpump + tax_df.loc[y, 'gen_tax'] * maxgen
        start_date = start_index.date()
        h_before = df.loc[start_index]
        days_of_year = df[df.year == (y+2000)].last_valid_index().dayofyear
        for d in range(0, days_of_year):
            day = start_date + dt.timedelta(d)
            next_day = day + dt.timedelta(days=1)


            # skip first hour of the year, iterate day by day
            for t, h in df[df.year == (y+2000)][:].loc[day.isoformat()].iterrows():

                # If pumping water 
                if (h.Price < h.Buy_price) & (h_before.Storage + maxpump < maxstorage):
                    h.Storage = h_before.Storage + maxpump*pump_eff
                    h.Pumping = maxpump*pump_eff
                    h.hourIncome = - maxpump*pump_eff
                    h.Income = h_before.Income - h.Price * maxpump
                    if y > 19:
                        h.Tax = h_before.Tax - (h.Price*maxpump + tax_df.loc[y, 'risk'])*tax_df.loc[y, 'loss_coeff']
                    else: 
                        h.Tax = h_before.Tax - maxpump*tax_df.loc[y, 'tarrif']
                    h.Generation = 0

                # If generating electricity
                elif (h.Price > h.Sell_price) & (h_before.Storage - maxgen > minstorage):
                    h.Storage = h_before.Storage - maxgen
                    h.Generation = maxgen*gen_eff
                    h.Income = h_before.Income + h.Price * h.Generation
                    h.hourIncome = h.Price * h.Generation
                    if y > 19:
                        h.Tax = h_before.Tax + (h.Price*maxgen + tax_df.loc[y, 'risk'])*tax_df.loc[y, 'loss_coeff']
                    else:
                        h.Tax = h_before.Tax + maxgen*tax_df.loc[y, 'tarrif']
                    h.Pumping = 0
                
                # If do nothing
                else:
                    cols = ['Storage', 'Total_generation', 'Tax', 'Income']
                    h.loc[cols] = h_before[cols]
                    h.Generation = 0
                    h.Pumping = 0
                    h.hourIncome = h.Price * h.Generation
                
                h.Total_generation = h_before.Total_generation + h.Generation
                df.loc[t] = h
                h_before = h

    res_df = df[(df['year'] -2000).isin(years)]
    res_df = res_df[res_df.Income.isna() == False] # to work for skottår
    res_df.loc[:, ['IncomeAfterTax']] = res_df.Income - res_df.Tax
    return res_df



def sell_curve(x, rate):
    sell_at_50_rate = 1.1
    # rate = 0.9*rate
    recent_mean = 1
    k = -2*(rate-1)*recent_mean
    # m = (2-rate)*recent_mean
    m = rate*sell_at_50_rate
    return k*x+m

def buy_curve(x, rate):
    buy_at_50_rate = 0.9
    recent_mean = 1
    k = 2*(rate-1)*recent_mean
    # m = (2-rate)*recent_mean
    m = buy_at_50_rate*(2 - rate)
    return k*x+m
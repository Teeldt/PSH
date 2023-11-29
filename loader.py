import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import datetime as dt
from smhi_open_data import SMHIOpenDataClient, Parameter

def load_plot_df(years=list(range(11,23)), weather='WindSpeed'):
    df = load_weather_price(years, weather)
    df["Hour"] = df.index.hour
    df["Weekday"] = df.index.weekday
    df["Month"] = df.index.month
    df["Year"] = df.index.year
    seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Autumn'}
    df['Season'] = df['Month'].replace(seasons)
    df.rename(mapper={'Price SE1': 'SE1', 'Price SE2': 'SE2', 'Price SE3': 'SE3', 'Price SE4': 'SE4'}, axis=1, inplace=True)
    water_s = load_water_levels()
    w_s = water_s.reindex(index=df.index).interpolate(method='linear').fillna(method='bfill')
    df[["Water_magasine", "Week"]] = w_s    
    fot_df = load_import_export(years)
    full_df = pd.merge(left=df, right=fot_df, how='inner', left_index=True, right_index=True)
    fcr_df = load_fcr_prices()
    full_df = full_df.merge(fcr_df, how='left', left_index=True, right_index=True)

    return full_df

def load_weather_price(years=[21], param="WindSpeed"):
    # main_df = load_more_prices(years)
    param_dict = {'TemperaturePast1h': 1,
                  'TemperaturePast24h': 2,
                  'WindSpeed': 4,
                  'PrecipPast24hAt06': 5,
                  'Humidity': 6,
                   'PrecipPast1h': 7,
                   'SunLast1h': 10,
                   'PrecipPast15m': 14,
                   'CloudCover': 16,
                   'PrecipPast12h': 17,
                   'WindSpeedTown': 21}

    # Winds
    client = SMHIOpenDataClient()
    stations = client.get_parameter_stations(parameter=Parameter[param])
    # used_stations = pd.DataFrame(columns=["Name", "id", "lat", "long"])

    # Make sure to run from exjobb-pumped-hydro root directory
    datapath = "weather/"+param+".pkl"
    try:
        df = pd.read_pickle(datapath)
    except:
        df = load_region_prices(list(range(11,24)))
        for s in stations:
            if not(s["active"]):
                continue
            else:
                s_id = str(s["id"])
                # used_stations.loc[len(used_stations.index)] = [s["name"], s["id"], s["latitude"], s["longitude"]]
                filepath = "data/smhi/smhi-opendata_" + str(param_dict[param]) +"_" + s_id + "_corrected-archive.csv"
                try:
                    temp_df = pd.read_csv(filepath, sep=";", header=10, usecols=[0,1,2], parse_dates=[[0,1]])
                except:
                    try:
                        urllib.request.urlretrieve("https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/" + str(param_dict[param]) +"/station/" + s_id + "/period/corrected-archive/data.csv", filepath)
                        temp_df = pd.read_csv(filepath, sep=";", header=10, usecols=[0,1,2], parse_dates=[[0,1]])
                    except:
                        print("no valid for station " + s_id)
                        continue
                
                col_name = s["name"]
                temp_df.columns = ["Times", col_name]
                # valid_times = wind_df["Times"].iloc[-1] > dt.datetime(2021, 12, 30)
                # if valid_times:
                try:
                    temp_df["Times"] = temp_df["Times"].dt.round("H")
                except:
                    print("fail at " + filepath)
                    continue
                temp_df = temp_df.set_index(["Times"])
                temp_s = temp_df[col_name]
                df = df.merge(temp_s, how='left', left_index=True, right_index=True)
        df = df[df.index.duplicated() == False]
        df.to_pickle(datapath)

    df[param] = df.loc[:, [x for x in df.columns if not("SE" in x)]].mean(numeric_only=True, axis=1)
    df = df.loc[:, ["SE1", "SE2", "SE3", "SE4", param]]
    df = df.loc[((df.index.year-2000).isin(years))]
    df["year"] = df.index.year
    return df
        
# def wind_price_plot(year=21):
#     df, _ = load_weather_price([year])
#     # pl = main_df.plot.scatter(y="Wind average", x="Price")
#     df['log'] = np.log(df["Weather average"])
#     # sns.scatterplot(df, x='Price', y='Weather average', label="Year " + str(2000+year))
#     sns.lmplot(df, y='Price', x='Weather average', logx=True)
#     # plt.set_title("Year " + str(2000+year))

#     # # trendline
#     z = np.polyfit(df.Price, df["Wind average"], 1)
#     p = np.poly1d(z)
#     plt.plot(df.Price, p(df.Price))
#     #plt.boxplot(x = main_df["Wind average"], y = main_df["Price"])
#     # plt.savefig("figures/smhi_wind_price_" + str(year) + ".eps")
#     # plt.close()
#     plt.show()

def load_prices(year=22, region = 'SE2'): # year from 11-22
    '''Load hourly energy prices from a specified year in SE2 and return as a pandas table'''
    cols_to_use = list(range(0, 4)) + list(range(5, 26))
    if year < 14: # different formatting
        cols_to_use = list(range(0, 25))
    hs = ["Day"] + list(range(1, 25))
    regions = {'SE1': 'lul', 'SE2': 'sund', 'SE3': 'sto', 'SE4': 'mal'}
    filename = "data/nordpool/"+region+"/"+ regions[region] + "sek" + str(year) + ".sdv"
    price_df = pd.read_csv(filename, sep=";", skiprows=6, usecols=cols_to_use, on_bad_lines = 'warn', parse_dates=[0], date_format='%d.%m.%y', header=None, names=hs, decimal=",", dtype=float, encoding='latin-1')
    p_df = price_df.set_index(["Day"]).stack()
    p_df = p_df.to_frame().reset_index()
    p_df.columns = ["Day", "Hour", "Price"]
    p_times = p_df["Day"] + p_df["Hour"] * pd.to_timedelta(pd.offsets.Hour(1).nanos)
    price_df = p_df["Price"].to_frame()
    price_df.index = p_times
    price_df = price_df.loc[(price_df.index.year == year+2000)]
    return price_df

def load_more_prices(years=list(range(11,23)), region='SE2'):
    price_df = load_prices(years[0], region)
    for y in years[1:]:
        price_df = pd.concat([price_df, load_prices(y, region)])
    return price_df

def load_import_export_year(year):
    # print(year)
    filename = 'data/svk/n_fot' + str(year+2000) + '-01-12.csv'
    if year == 23:
        filename = 'data/svk/n_fot' + str(year+2000) + '-01-07.csv'
    cols = ['Time', 'Total Consumption', 'Wind Production', 'Water Production', 'Nuclear Production', 'Gas/Diesel production', 'Heat Production', 'Unspecified Production', 'Solar Production', 'Total production', 'Import/Export']
    
    if year < 12:
        skip = 5
        end=2
    elif year > 16:
        cols.remove('Gas/Diesel production')
        skip = 7
        end = 0
    else:
        skip = 7
        end = 0

    fot_df = pd.read_csv(filename, skiprows = skip, header=None, delimiter=',', encoding='utf-8', parse_dates=[0], date_format = '%d.%m.%Y %H:%M', dtype=float)
    # fot_df = fot_df[order]
    if end:
        fot_df = fot_df.iloc[:-end, :]
    fot_df.columns = cols
    fot_df.index = pd.to_datetime(fot_df['Time'], format = '%d.%m.%Y %H:%M')
    # fot_df.index = fot_df.Time
    fot_df = fot_df.iloc[:, 1:]
    return fot_df

def load_import_export(years=list(range(11,23))):
    df = load_import_export_year(years[0])
    for y in years[1:]:
        df = pd.concat([df, load_import_export_year(y)])
    return df

def load_region_prices(years=list(range(11,23))):
    regions = ['SE2', 'SE3', 'SE4']
    df = load_more_prices(years, 'SE1')
    df.rename(mapper = {"Price": 'SE1'}, axis=1, inplace=True)
    for r in regions:
        next_df = load_more_prices(years, r)
        next_df.rename(mapper = {"Price": r}, axis=1, inplace=True)
        df = df.merge(next_df, left_index=True, right_index = True)
    return df

def load_fcr_prices():
    filename1 = 'data/fcr/FCR_from2021.csv'
    prices1 = pd.read_csv(filename1, header=0, delimiter=';', encoding='utf-8', parse_dates=[0], date_format = '%Y-%m-%d %H:%M:%S', decimal=',', skipfooter=1, engine='python')
    cols1 = ['Datum', 'FCR-D upp Pris (EUR/MW)', 'FCR-D ned Pris (EUR/MW)']
    p1 = prices1[cols1].set_index(['Datum'])

    filename2 = 'data/fcr/FCR_to2020.csv'
    prices2 = pd.read_csv(filename2, header=0, delimiter=';', encoding='utf-8', parse_dates=[0], date_format = '%Y-%m-%d %H:%M', decimal=',', skipfooter=1, engine='python')
    cols2 = ['Period', 'FCR-D upp Pris (EUR/MW)', 'FCR-D ned Pris (EUR/MW)']
    p2 = prices2[cols2].set_index(['Period'])

    p = pd.concat([p2, p1])

    exchange = pd.read_csv('data/fcr/EURtoSEK.csv', header=0, delimiter=';', encoding='utf-8', parse_dates=[0], date_format = '%Y-%m-%d %H:%M', decimal=',', skipfooter=1, engine='python')
    c = ['Period', 'Värde']
    e = exchange[c].set_index(['Period']).iloc[::-1]
    e.columns = ['SEK/EUR']
    df = p.merge(e, how='left', left_index=True, right_index=True).fillna(method='ffill')

    df.rename(mapper={'FCR-D upp Pris (EUR/MW)': 'FCR-D_up',
               'FCR-D ned Pris (EUR/MW)': 'FCR-D_down'}, axis=1, inplace=True)
    df['SEK/EUR'].fillna(method='bfill', inplace=True)
    df['FCR-D_up'].fillna(value=0, inplace=True)
    df['FCR-D_down'].fillna(value=0, inplace=True)
    df['FCR-D_up'] = df['FCR-D_up']*df['SEK/EUR']
    df['FCR-D_down'] = df['FCR-D_down']*df['SEK/EUR']

    # df.to_csv('fcr-price.csv')

    return df


def load_water_level(year=22):
    '''Load weekly water magasine levels for year and return as table.'''
    col_to_use = [0, year-9]
    filename = "data/Magasin_2010-2022.xlsx"
    m_df = pd.read_excel(filename, skiprows=4, usecols=col_to_use)
    m_df.columns = ["Date", "Level"]
    m_df['Week'] = m_df['Date']
    w = dt.date.fromisocalendar(2000+year, 1, 3) # 3 for wednesday when it's updated
    w_start = dt.datetime(w.year, w.month, w.day)
    p_times = w_start + (m_df["Date"] - 1) * pd.to_timedelta(pd.offsets.Day(n=7).nanos)
    m_df.index = p_times
    m_s = m_df[["Week", "Level"]]
    m_s = m_s.dropna()
    return m_s

def load_water_levels():
    years=list(range(11,23))
    df = load_water_level(years[0])
    for y in years[1:]:
        df = pd.concat([df, load_water_level(y)])
    df = df.loc[df.index.notnull()]
    return df

def load_taxes():
    # Taxes for Juktan
    y = list(range(11,24))
    years = list(range(2011,2024))
    tax_df = pd.DataFrame(columns=['gen_tax', 'pump_tax', 'loss_coeff', 'tarrif', 'risk'], index=y)
    input = [41, 42, 45, 45, 45, 45, 45, 49, 57, 69, 70, 71, 28] # unit SEK/kW
    output = [31, 46, 51, 51, 51, 55, 55, 61, 71, 86, 91, 90, 36] # unit SEK/kW
    loss = [5, 5, 5, 6, 6, 5, 5, 5.5, 5.8, 5.7, 5.8, 6, 6.5] # unit %
    tarrif = [19, 20, 18.2, 18.96, 18.24, 13.8, 10.96, 12.1, 13.46] #t.o.m. 2018 # unit SEK/MWh
    risk = [10, 10, 10, 11] #unit SEK/MWh
    tax_df.gen_tax = [input[i]*1000 for i in range(len(years))] # unit SEK/MW
    tax_df.pump_tax = [output[i]*1000 for i in range(len(years))] # unit SEK/MW
    tax_df.loss_coeff = [loss[i]/100 for i in range(len(years))] # proportion
    tax_df.loc[11:19, 'tarrif'] = [tarrif[i] for i in range(len(tarrif))] # unit SEK/MWh
    tax_df.loc[20:23, 'risk'] = [risk[i] for i in range(len(risk))]  # unit SEK/MWh
    return tax_df
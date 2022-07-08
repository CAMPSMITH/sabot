"""
Helper Function for importing chart data from Crypto Compare API
"""
import pandas as pd
import numpy as np
import requests
import json
import os
# from dotenv import load_dotenv
import math

from datetime import datetime,timezone
import pytz


def epoch_to_datetime(epoch_time,time_zone_name='US/Eastern'):
    """
    Converts unix epoch time to date time
    If a time zone is provided, it will convert to that timezone.
    By default, time will be returned in US/Eastern time.
    If time_zone_name is set to None, it will return UTC time.
    This function is written to be imported into a notebook or other python program.
    If using Pandas, this function can be used in apply()
    Parameters:
        epoch_time (int): epoch time, must be an int
        time_zone_name (str): time zone name to localize time to, e.g. `US/Eastern`
    """

    epoch_time = int(epoch_time) # coerce to int in case is came in as a string
    dt = datetime.utcfromtimestamp(epoch_time) 
    dt = dt.replace(tzinfo=timezone.utc) # coerce to UTC for proper time zone conversion
    # return dt.astimezone(pytz.timezone(time_zone_name))
    return dt


def crypto_compare_chart(fsym,
                         tsym,
                         limit,
                         toTs,
                         api_key,
                         resolution='histohour'
                        ):
    """
    Provide a ticker symbol and a valid API key for Crypto Compare and a corresponding dataframe will be returned.
    
    Parameters:
    
    fsym (str) : ticker to get closing data
    tsym (str) : base currency
    limit (int) : number of cycles out to retrieve data
    toTs (int) : start date in epoch time
    api_key (str) : retrieve api key from Crypto Compare (https://min-api.cryptocompare.com/)
    resolution (str) : ='histohour' the time interval to retrieve data
    
    Returns:
    
    response:   (json) The json response object
    
    """
    #Base URL for API
    base_url = "https://min-api.cryptocompare.com/data/v2/"
    
    url_query=base_url+resolution
    
    querystring = {
        "fsym":fsym,
        "tsym":tsym,
        "limit":limit,
        "toTs":toTs
    }

    headers = {
        'api_key': api_key
        }

    response = requests.request("GET", url_query,
                                params=querystring,
                                headers=headers).json()  
    return response



def number_of_iterations(start_date, end_date):
    if start_date==-1:
        start_date = datetime.now()
    else:
        start_date = datetime.strptime(start_date, "%d/%m/%Y %H:%M:%S")
        
    end_date = datetime.strptime(end_date, "%d/%m/%Y %H:%M:%S")
    
    ts= (start_date - end_date).total_seconds()
    
    th = ts/3600
    
    full_iterations = math.floor(th/2000)
    
    last_iteration = math.ceil((th-full_iterations*2000))
    
    return (start_date.timestamp(),
            end_date.timestamp(),
            full_iterations, 
            last_iteration)


def to_dataframe(time,
                 open_price,
                 high_price,
                 low_price,
                 close_price,
                 volume_from,
                 volume_to):
    """
    Provide seven list objects (time in epoch time), open, high, low, close, volume from, and volume to.  Will convert to dataframe with time coverted to UTC time.
    
    Parameters:
    
    time (list) : epoch time array

    open_price (list) : close prices
    high_price (list) : close prices
    low_price (list) : close prices    
    close_price (list) : close prices
    volume_from (list) : close prices
    volume_to (list) : close prices
   
    Returns:
    
    df_price : A seven column dataframe with time in UTC and all ohlcvv prices.
    
    """
    time = pd.Series(time)
    
    open_price = pd.Series(open_price)
    high_price = pd.Series(high_price)
    low_price = pd.Series(low_price)
    close_price = pd.Series(close_price)    
    volume_from = pd.Series(volume_from)
    volume_to = pd.Series(volume_to)
   
    df_price = pd.DataFrame({'Epoch Time':time,
                             'open': open_price,
                             'high': high_price,
                             'low': low_price,
                             'close':close_price,
                             'volume_from':volume_from,
                             'volume_to':volume_to})   
    
    df_price['Time (UTC)'] = df_price['Epoch Time'].apply(lambda x: epoch_to_datetime(x, time_zone_name=None))
    df_price.index = df_price['Time (UTC)']
    df_price.drop(columns=['Epoch Time','Time (UTC)'],inplace=True)
    
    return df_price


def get_historical_prices(currency,
                          base_currency,
                          end_date,
                          api_key,
                          start_date=-1):
    """
    Provide the crypto currency, the base currency of comparison, the start date, and the end date of the time period.  Calls CryptoCompare API and returns a dataframe with at least as many hourly datapoints for the specified end date.  The default value for the start_date is the present time (-1).
    
    Parameters:
    
    currency (string) : currency to obtain
    base_currency (string) : currency pair to get price compare
    end_date (datetime string) : time period to end API calls
    api_key (string) : API key from CryptoCompare.com
    start_date (datetime string) : (=-1 default) Beginning date to begin call. A value of -1 is the present time.    
   
    Returns:
    
    df_historical_prices : A seven column dataframe with time in UTC and all ohlcvv prices.
    
    """
    #Initialize all lists for data
    time=[]
    open_price=[]
    high_price=[]
    low_price=[]
    close_price=[]
    volume_from=[]
    volume_to=[]
    
    start_date, end_date, api_calls, last_request = number_of_iterations(-1, end_date)
    
    #Make the API calls
    for i in range(api_calls+1):
        temp_time=[]

        temp_open_price=[]
        temp_high_price=[]
        temp_low_price=[]
        temp_close_price=[]
        temp_volume_from=[]
        temp_volume_to=[]

        if(i < api_calls):
            request_frames = 2000
        else:
            request_frames = last_request

        response = crypto_compare_chart(currency,
                                   base_currency,
                                   request_frames,
                                   start_date,
                                   api_key)

        price_list = response['Data']['Data']

        for entry in price_list:
            temp_time.append(entry['time'])

            temp_open_price.append(entry['open'])
            temp_high_price.append(entry['high'])
            temp_low_price.append(entry['low'])
            temp_close_price.append(entry['close'])
            temp_volume_from.append(entry['volumefrom'])
            temp_volume_to.append(entry['volumeto'])


        time[:0] = temp_time
        open_price[:0]=temp_open_price
        high_price[:0]=temp_high_price
        low_price[:0]=temp_low_price
        close_price[:0]=temp_close_price
        volume_from[:0]=temp_volume_from
        volume_to[:0]=temp_volume_to  

        print(f'Appended API request #{i} to dataframe')
        start_date=time[0] - 3600
   
    #Create dataframe
    df_historical_price = to_dataframe(time,
                                         open_price,
                                         high_price,
                                         low_price,
                                         close_price,
                                         volume_from,
                                         volume_to)
    return df_historical_price


"""
Helper Function for importing chart data from Crypto Compare API
"""
import pandas as pd
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
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
    if time_zone_name is None:
        return dt # return UTC time\n",
    # time zone set\n",
    return dt.astimezone(pytz.timezone(time_zone_name))




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


def to_dataframe(time, close):
    """
    Provide two list objects (time in epoch time) and close price.  Will convert to dataframe with time coverted to UTC time.
    
    Parameters:
    
    time (list) : epoch time array
    close (list) : close prices
   
    Returns:
    
    df_price : A two column dataframe with time in UTC and close price.
    
    """
    time = pd.Series(time)
    close= pd.Series(close)
    df_price = pd.DataFrame({'Epoch Time':time,
                        'Close Price':close})
    df_price['Time (UTC)'] = df_price['Epoch Time'].apply(lambda x: epoch_to_datetime(x, time_zone_name=None))
    df_price.index = df_price['Time (UTC)']
    df_price.drop(columns=['Epoch Time','Time (UTC)'],inplace=True)
    
    return df_price
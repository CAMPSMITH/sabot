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

from datetime import datetime,timezone,timedelta
import pytz

global cryptocompare_apikey
load_dotenv()
cryptocompare_apikey = os.getenv('CRYPTO_COMPARE_API_KEY')


"""
###CryptoCompare
CRYPTO_COMPARE_API_KEY= "1cbd3ca59b2a9fcca8452ce7a0c52113319c0a7bb574a471a1668b29cf46c139"
"""


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

def get_recent_prices(currency,
                      base_currency,
                      end_date,
                      cryptocompare_apikey,
                      start_date=-1):
    """
    A slightly modified version of the get_historical_prices function.  Calls CryptoCompare API and returns a dataframe with at least as many hourly datapoints for the specified end date.  The default value for the start_date is the present time (-1).
    
    Parameters:
    
    currency (string) : currency to obtain
    base_currency (string) : currency pair to get price compare
    end_date (datetime string) : time period to end API calls
    api_key (string) : API key from CryptoCompare.com
    start_date (datetime string) : (=-1 default) Beginning date to begin call. A value of -1 is the present time.    
   
    Returns:
    
    recent_prices : A dictionary containing the currency as a key to a dictionary of time, open, high, low, close, volume prices.
    
    """
    #Initialize all lists for data
    time=[]
    open_price=[]
    high_price=[]
    low_price=[]
    close_price=[]
    volume=[]
    
    start_date, end_date, api_calls, last_request = number_of_iterations(-1, end_date)
    
    #Make the API calls
    for i in range(api_calls+1):
        temp_time=[]

        temp_open_price=[]
        temp_high_price=[]
        temp_low_price=[]
        temp_close_price=[]
        temp_volume=[]

        if(i < api_calls):
            request_frames = 2000
        else:
            request_frames = last_request

        response = crypto_compare_chart(currency,
                                   base_currency,
                                   request_frames,
                                   start_date,
                                   cryptocompare_apikey)

        price_list = response['Data']['Data']

        for entry in price_list:
            temp_time.append(entry['time'])

            temp_open_price.append(entry['open'])
            temp_high_price.append(entry['high'])
            temp_low_price.append(entry['low'])
            temp_close_price.append(entry['close'])
            temp_volume.append(entry['volumeto'])


        time[:0] = temp_time
        open_price[:0]=temp_open_price
        high_price[:0]=temp_high_price
        low_price[:0]=temp_low_price
        close_price[:0]=temp_close_price
        volume[:0]=temp_volume  

        print(f'Appended API request #{i} to dataframe')
        start_date=time[0] - 3600
   
    #Create dictionary
    recent_prices = {
        currency:pd.DataFrame({
            'time':list(map(epoch_to_datetime, time)),
            'open':open_price,
            'high':high_price,
            'low':low_price,
            'close':close_price,
            'volume':volume
        })
    }

    return recent_prices


def initialize_prices(sma_window, coin_list, cryptocompare_apikey, base_currency='USD'):
    '''
    Uses CryptoCompare's API to create a large dictionary of ohlcv recent prices of a list of crypto coins
    
     Parameters:
    
    sma_window (int) : window of time in hours for pricing to be found
    coin_list (list) : a list containing strings of coins to find pricing data (ex: ['BTC','ETH','USDT'])
    api_key (string) : API key from CryptoCompare.com
    base_currency (string) : (='USD' default) Base currency for pricing data.    
   
    Returns:
    
    ohlcv_prices (dict) : A dictionary containing all ohlcv pricing data for all currencies.
    '''
    
    
    # Initialize date request
    current_date = datetime.now()
    end_date = current_date - timedelta(hours=sma_window)
    end_date = end_date.strftime('%d/%m/%Y %H:%M:%S')

    # Initialize Dictionary
    ohlcv_prices = {}
    
    for coin in coin_list:
        
        historical_prices = get_recent_prices(coin,
                                              base_currency,
                                              end_date,
                                              cryptocompare_apikey)
        ohlcv_prices.update(historical_prices)
    
    return ohlcv_prices



def update_prices(ohlcv_prices, api_key):

    coin_list=list(ohlcv_prices.keys())
    
    price_data=list(ohlcv_prices[coin_list[0]].keys())
    
    new_prices=initialize_prices(0,
                                 coin_list,
                                 api_key)
    
    last_update_time = ohlcv_prices[coin_list[0]]['time'][-1]
    current_update_time = new_prices[coin_list[0]]['time'][-1]
    
    if(current_update_time ==last_update_time):
        print('Already Up to Date')
        return ohlcv_prices
    else:

        for coin in coin_list:
            for price in price_data:
                # Pop off oldest entries in buffer
                ohlcv_prices[coin][price].pop(0)
                                  
                # Append newest prices in buffer
                ohlcv_prices[coin][price].append(new_prices[coin][price][-1])
        
        # Return buffer
        return ohlcv_prices
        

        
def get_cryptocompare_ohlcv(currency, sma_window, df=None):
    global cryptocompare_apikey
    
    if df is None:
        df = pd.DataFrame()
    
    sma_window=sma_window-2 
    
    rows = df.shape[0]
    if rows > 1:

        updated_entries = sma_window - rows +2
        df_append = initialize_prices(updated_entries,
                       [currency],
                       cryptocompare_apikey)[currency]
        if(df_append.iloc[-1,0] == df.iloc[-1,0]):
            print("dataframe is up-to-date")
        else:
            df = df.iloc[1: , :]
            print("popping oldest record")


            df = df.append(df_append.iloc[-1,:],
                          ignore_index=True)
            print(f"getting {updated_entries+1} {currency} records")
    else:
        df = initialize_prices(sma_window,
                           [currency],
                           cryptocompare_apikey)[currency]
        print(f"getting {sma_window+2} {currency} records")
    
    
    return df # should be in a normalized ohlcv for this currency



def transaction_quote(buy_token, sell_token, sell_amount):
    #Ensure correct tokens to ensure correct decimals
    
    # check if sell token is ok. If not, just print an error and do nothing
    if sell_token != 'SUSD' and sell_token != 'sUSD' and sell_token != 'USDT':
        print("Sell token not equal to SUSD or USDT. Re-set and try again.")
         
    else:  
        # next, check the buy and if that's not good, also print and do nothing
        if buy_token != 'SUSD' and buy_token != 'sUSD' and buy_token != 'USDT':
            print("Buy token not equal to SUSD or USDT. Re-set and try again.")
        
        else:
            # if we got here, buy and sell are good and so now we do work
            if sell_token == 'SUSD'or sell_token =='sUSD':
                token_decimals = 1000000000000000000

            if sell_token == 'USDT':
                token_decimals = 1000000

            decimalized_sell_amount = sell_amount * token_decimals

            #get response
            response = requests.get(f'https://api.0x.org/swap/v1/quote?buyToken={buy_token}&sellToken={sell_token}&sellAmount={decimalized_sell_amount}')
            response = response.json()
            
            #print(response)
            #Calculate gas cost and pricing
            gas_units = float(response['gas'])
            gas_price = float(response['gasPrice'])
            total_gas_cost = gas_units*gas_price
            percentage_of_eth = float(total_gas_cost/1000000000000000000)
            gas_cost_in_stablecoin = float(response['buyTokenToEthRate'])*(float(percentage_of_eth))
            price = response['price']
            guaranteed_price = response['guaranteedPrice']
            
           # buy_token_multiplied_by_price = float(buy_token) * float(price)
            
            #output = float(price) * float(buy_token)
            
            #need to see total amount of tokens swapped
            #print(f'{sell_amount}{sell_token} = {buy_token_multiplied_by_price}')
            #print(f'Price: {sell_amount} {sell_token} = {output} {sell_token}')
            
            x = float(price)*float(sell_amount)
            guaranteed_x = float(guaranteed_price)*float(sell_amount)
            
            
            print(f'Conversion Price: {price}')
            print(f'Total Swap: {sell_amount} {sell_token} = {x} {buy_token}')
            print(f'Guaranteed Swap: {sell_amount} {sell_token} = {guaranteed_x} {buy_token}')
            print(f'Gas Cost: $',(gas_cost_in_stablecoin))
            return x,gas_cost_in_stablecoin
    return
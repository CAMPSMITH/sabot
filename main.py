"""Sabot trading bot

on invocation

1 - get data for prediction
2 - get prediction / decision from machine learning model (inference engine)
4 - if trade is indicated
5 - get balance from wallet
6 - if funds available, execute trade
7 - if purchase executed, deduct wallet
8 - if sale executed, increment wallet

"""
import sys
import os
import json
from wallet import Wallet
import fire
import pickle
from pathlib import Path
import pandas as pd
from datetime  import datetime
import schedule
from dotenv import load_dotenv
from utils.utils import epoch_to_datetime, transaction_quote
load_dotenv()
import json
import requests
import csv
from utils.eval import add_engineered_features
import ccxt
exchange = ccxt.kucoin()
 
# ------ global references 
# define data sources
sources = [
    {"currency":"BTC","api":"cryptocompare"},
    {"currency":"ETH","api":"cryptocompare"},
    {"currency":"USDT","api":"cryptocompare"},
    {"currency":"sUSD/USDT","api":"kucoin"}
]

wallet = None
model = None
x_scaler = None
ohlcv_cache = None
cryptocompare_apikey = None
kucoin_apikey = None
results_path = 'results/sabot.csv'

is_first_activation = True
csv_columns = ['time','decision','portfolio value','gas','close']

def push(update):
    global is_first_activation
    # on first activation, reset the results file
    # otherwise, simpley append the results
    if is_first_activation:
        with open(results_path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(update)
        is_first_activation = False
    else:
        with open(results_path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(update)


def calculate_effective_flat_fee(amount, pct_fee, flat_fee):
    return  amount * pct_fee + flat_fee

def calculate_effective_pct_fee(amount, pct_fee, flat_fee):
    return  (amount * pct_fee + flat_fee) / amount

    
def swap(txn,max_trade_amount):
    global wallet

    fee_cap_flat = 140
    fee_cap_pct = 0.0015
    pct_fee = 0.0004
    
    # check fees
    try:
        target_amount , flat_fee = transaction_quote(buy_token=txn['target_currency'], sell_token=txn['source_currency'], sell_amount=txn['source_amount'])
        effective_flat_fee = calculate_effective_flat_fee(txn['source_amount'],pct_fee,flat_fee)
        effective_pct_fee = calculate_effective_pct_fee(txn['source_amount'],pct_fee,flat_fee)
        assert effective_flat_fee <= fee_cap_flat, f"amount: {txn['source_amount']}, pct fee: {pct_fee}, flat fee: {flat_fee}, effective flat fee: {effective_flat_fee} exceeds flat fee cap of {fee_cap_flat}, ignoring txn"
        assert effective_pct_fee <= fee_cap_pct, f"amount: {txn['source_amount']}, pct fee: {pct_fee}, flat fee: {flat_fee}, effective pct fee: {effective_pct_fee} exceeds pct fee cap of {fee_cap_pct}, ignoring txn"

        # assume fees are paid from source account
        # reduce target_amount by 4bps
        source_amount = txn['source_amount']
        if source_amount > max_trade_amount:
            source_amount = max_trade_amount
        if source_amount > wallet.get_balance(txn['source_currency']) - effective_flat_fee:
            source_amount = wallet.get_balance(txn['source_currency']) - effective_flat_fee
        assert source_amount > 0.00, f"pre trade source amount is {source_amount}, ignoring transaction"
        print(f"swapping {source_amount} {txn['source_currency']} into {txn['target_currency']}, total fees: {effective_flat_fee} into {target_amount} {txn['target_currency']}")
        wallet.withdraw(txn['source_currency'],source_amount + flat_fee)
        wallet.deposit(txn['target_currency'],target_amount)
        return target_amount,flat_fee 

    except AssertionError as err:
        print(err)
    
    return None,None # buyamount,gas

def get_cryptocompare_ohlcv(currency,sma_window):

    #Base URL for API
    base_url = "https://min-api.cryptocompare.com/data/v2/"
    resolution='histohour'

    url_query=base_url+resolution

    querystring = {
        "fsym":currency,
        "tsym":"USD",
        "limit":sma_window+3,
        "toTs":datetime.now().replace(minute=0,second=0).timestamp()
    }

    headers = {
        'api_key': cryptocompare_apikey
        }

    response = requests.request("GET", url_query,
                                params=querystring,
                                headers=headers).json()  
    assert "Data" in response, f"crypto compare response is missing Data element"
    assert "Data" in response["Data"], f"crypto compare response is missing Data element"
    df = pd.DataFrame.from_records(response["Data"]["Data"])
    df['time'] = df['time'].apply(epoch_to_datetime)
    df.drop(columns=['volumefrom','conversionType', 'conversionSymbol'],inplace=True)
    df.set_index('time',inplace=True)
    df.rename(columns={"volumeto": "volume"},inplace=True)
    df = df.iloc[-sma_window:,]
    return df    

def get_kucoin_ohlcv(currency,sma_window):
    global exchange
    df = pd.DataFrame(exchange.fetchOHLCV(currency.upper(), timeframe = "1h", limit = sma_window + 5, params={'price':'index'}))
    df.columns = ['epoch', 'open', 'high', 'low', 'close', 'volume']
    df['epoch'] = df['epoch']/1000  # from epoch in ms to epoch in seconds
    df['time'] = df['epoch'].apply(epoch_to_datetime)
    df.drop(columns=['epoch'], inplace = True)
    df.set_index('time', inplace = True)
    df = df.iloc[-sma_window:,]
    # print(df)    
    return df

def get_prediction(X_scaled):
    global model
    return model.predict(X_scaled)

def build_txn(y, txn_max):
    if y==0:
        return None      
    if y==1:
        return {
         "txn":"buy", 
         "source_currency":"USDT",
         "source_amount":txn_max,
         "target_currency":"sUSD",
        }
    if y==-1:
        return {
         "txn":"sell", 
         "source_currency":"sUSD",
         "source_amount":txn_max,
         "target_currency":"USDT",
        }

# prescaled_X = None    
# choices = [0,1,2]
# def get_X():          
#     global prescaled_X
#     # a mock that generates prescaled X records
#     if prescaled_X is None:
#         prescaled_X = pd.read_csv(Path('data/unscaled_X.csv'))
#     i = random.choice(choices)
#     x_series = prescaled_X.iloc[i,:]
#     df = x_series.to_frame().T
#     df['Time (UTC)'] = datetime.now()
#     df = df.set_index('Time (UTC)')
#     print(f"selected record {i}, y={df['y_pred'].iloc[0]}")
#     df = df.drop(columns=['y_pred'])
#     return df

def on_trigger(fast_sma_window,slow_sma_window, txn_max, factor):
    global model
    global x_scaler
    global wallet
    global ohlcv_cache

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sabot activating")

    ohlcv_data = []

    usdt_close = None

    for source in sources:
        currency = source["currency"]
        # on activation, first get the latest ohlcv data
        if source["api"] == "cryptocompare":
            ohlcv = get_cryptocompare_ohlcv(currency, slow_sma_window)
        if source["api"] == "kucoin":
            ohlcv = get_kucoin_ohlcv(currency, slow_sma_window)
        ohlcv = ohlcv.reindex(['open','high','low','close','volume'], axis=1)
        if currency == "USDT":
            usdt_close = ohlcv['close'].iloc[-1]
        # next add engineered features and prep for concatenation
        ohlcv = add_engineered_features(ohlcv,fast_sma_window,slow_sma_window)

        # prefix column names so they can be differentiated
        cols =  ohlcv.columns
        new_cols = []
        for col in cols:
            new_cols.append(f"{currency}_{col}")                                        
        ohlcv.columns=new_cols
        ohlcv_data.append(ohlcv)

    df = pd.concat(ohlcv_data,axis=1)
    df['sUSD/USDT_offset'] = df['sUSD/USDT_std'] * factor
    # add actual returns
    df['returns'] = df['sUSD/USDT_close'].pct_change()
    # print(df.columns)
    # scale input data
    X_scaled = x_scaler.transform([df.iloc[-1,:]])

    # get prediction
    y = get_prediction(X_scaled)
    print(f"y = {y}")

    # build txn

    txn = build_txn(y, txn_max)

    if txn is None:
        print(f"no buy or sell.  y={y} ")
        print(str(wallet))  
        push({
            "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "decision":y[0],
            "portfolio value": wallet.get_total_value(),
            "gas":None,
            "close":usdt_close
        })     
        return
    amount, gas = swap(txn,txn_max)
    wallet.save()
    print(str(wallet))
    push({
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "decision":y[0],
        "portfolio value": wallet.get_total_value(),
        "gas":gas,
        "close":usdt_close
    })     


def run(model_file, 
        scaler_file,
        wallet_id,
        txn_max,
        fast_sma_window,
        slow_sma_window,
        factor
       ):
    global model
    global x_scaler
    global wallet

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sabot starting")

    # ---  Set up paper trading components
    # load model and pickle file
    model_path = Path(model_file)
    scaler_path = Path(scaler_file)
    assert model_path.exists(), f"{model_file} not found"
    assert scaler_path.exists(), f"{scaler_file} not found"
    model = pickle.load(open(model_path,'rb'))
    x_scaler = pickle.load(open(scaler_path,'rb')) 
    assert model is not None, f"unable to instantiate model"
    assert x_scaler is not None, f"unable to instantiate x_scaler"

    # initialize wallet
    wallet = Wallet(wallet_id)   
    assert wallet is not None, f"unable to instantiate wallet"

    # Configure and load scheduler
    # scheduler should call on_trigger()

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sabot ready")

    schedule.every().hour.at(":01").do(on_trigger, sma_window=slow_sma_window, txn_max=txn_max,factor=factor)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sabot terminating")

if __name__ == "__main__":
    fire.Fire(run)

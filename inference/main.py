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

import random   # TODO: remove when integrating model
 
# ------ global references 
# coin names for ordering purposes
currencies=['BTC','ETH','USDT','sUSD/USDT']
wallet = None
model = None
x_scaler = None
ohlcv_cache = None
cryptocompare_apikey = None
kucoin_apikey = None

def get_data():
    """ get data needed to get a prediction
    data needs to be scaled properly
    """
    print("getting_data")
    return {}

def get_prediction(x_scaled):
    """ get data needed to get a prediction
    data needs to be scaled properly
    """
    y_predict = 1
    print(f"model prediction {y_predict} for data {x_scaled}")
    return y_predict

def calculate_effective_flat_fee(amount, pct_fee, flat_fee):
    return  amount * pct_fee + flat_fee

def calculate_effective_pct_fee(amount, pct_fee, flat_fee):
    return  (amount * pct_fee + flat_fee) / amount

def get_target_amount_fees(source_currency, source_amount, target_currency, txn=None):
    # TODO: get fee details
    # injected in txn for now
    return txn['target_amount'],txn['flat_fee'], txn['flat_pct']
    
def swap(txn,max_trade_amount):
    global wallet
    # assume pct_fee is 0.0005 based on curve
    # 
    # get_fee(source_currency, target_currency, source_amount)

    fee_cap_flat = 140
    fee_cap_pct = 0.0015
    print(f"txn: {txn}")

    # mock for transaction fees
    target_amount,flat_fee, pct_fee = get_target_amount_fees(txn['source_currency'],txn['source_amount'],txn['target_currency'],txn)
    
    # check fees
    try:
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

    except AssertionError as err:
        print(err)
        return
    
def get_cryptocompare_ohlcv(currency, sma_window, df=None):
    global cryptocompare_apikey
    if df is None:
        df = pd.DataFrame()
    rows = df.shape[0]
    if rows > 1:
        print("popping oldest record")
    print(f"getting {sma_window-rows} {currency} records")
    return df # should be in a normalized ohlcv for this currency

def get_kucoin_ohlcv(currency,sma_window,df=None):
    global kucoin_apikey
    if df is None:
        df = pd.DataFrame()
    rows = df.shape[0]
    if rows > 1:
        print("popping oldest record")
    print(f"getting {sma_window-rows} {currency} records")

    # datetime.now().replace ... 0 out minute seconds, microseconds
    # get timestamp from ^^^ -> epoch in seconds
    # multiply ^^^ 1000 , cctx uses epoch time in milli
    # see Yanick;s
    # see eval.oy where we prep kucoin
    return df # should be in a normalized ohlcv for this currency

def get_prediction(X_scaled):
    global model
    return model.predict(X_scaled)

def build_txn(y, txn_max):
    if y==0:
        return None
    

    source_amount = txn_max                    # TODO: remove when trade info API is integrated
    r = random.random()                        # TODO: remove when trade info API is integrated
    r = r - 0.5 # shifted to be + or -         # TODO: remove when trade info API is integrated
    factor = r / 0.5 / 100 + 1                 # TODO: remove when trade info API is integrated
    expected_target_amount = txn_max * factor  # TODO: remove when trade info API is integrated
      
    if y==1:
        return {
         "txn":"buy", 
         "source_currency":"USDT",
         "source_amount":source_amount,
         "target_currency":"sUSD",
         "target_amount":expected_target_amount,   # TODO - remove when trade info api is integrated
         "flat_fee":9.00,   # TODO - remove when trade cost API is integrated
         "flat_pct":0.0010            # TODO - remove when trade cost API is integrated
        }
    if y==-1:
        return {
         "txn":"sell", 
         "source_currency":"sUSD",
         "source_amount":source_amount,
         "target_currency":"USDT",
         "target_amount":expected_target_amount,   # TODO - remove when trade info api is integrated
         "flat_fee":9.00,   # TODO - remove when trade cost API is integrated
         "flat_pct":0.0010            # TODO - remove when trade cost API is integrated
        }

prescaled_X = None    # TODO: remove when source data is integrated
choices = [0,1,2]
def get_X():          # TODO: remove when source data is integrated
    global prescaled_X
    # a mock that generates prescaled X records
    if prescaled_X is None:
        prescaled_X = pd.read_csv(Path('data/unscaled_X.csv'))
    i = random.choice(choices)
    x_series = prescaled_X.iloc[i,:]
    df = x_series.to_frame().T
    df['Time (UTC)'] = datetime.now()
    df = df.set_index('Time (UTC)')
    print(f"selected record {i}, y={df['y_pred'].iloc[0]}")
    df = df.drop(columns=['y_pred'])
    return df

def on_trigger(sma_window, txn_max):
    global model
    global x_scaler
    global wallet
    global ohlcv_cache

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sabot activating")

    # on activation, first get the latest ohlcv data
    currency = "BTC"
    ohlcv_cache[currency] = get_cryptocompare_ohlcv(currency, sma_window,ohlcv_cache[currency])
    currency = "ETH"
    ohlcv_cache[currency] = get_cryptocompare_ohlcv(currency, sma_window,ohlcv_cache[currency])
    currency = "USDT"
    ohlcv_cache[currency] = get_cryptocompare_ohlcv(currency, sma_window,ohlcv_cache[currency])
    currency = "sUSD/USDT"
    ohlcv_cache[currency] = get_kucoin_ohlcv(currency, sma_window,ohlcv_cache[currency])

    # next add engineered features and prep for concatenation
    ohlcv_data = []
    for currency in currencies:
        print(f"adding engineering data to {currency} and preparing to concetenate into X")
        ohlcv_data.append(ohlcv_cache[currency])
        # fix column names

    df = pd.concat(ohlcv_data)

    # add actual returns
    # df['return'] = df['sUSD/USDT_close'].pct_change()

    # scale input data
    df = get_X()  # TODO: remove this when integrating source data is done.  this is  mock source of prescaled X data
    X_scaled = x_scaler.transform([df.iloc[-1,:]])

    # get prediction
    y = get_prediction(X_scaled)
    print(f"y = {y}")

    # build txn

    txn = build_txn(y, txn_max)

    if txn is None:
        print(f"no buy or sell.  y={y} ")
        print(str(wallet))
        return
    swap(txn,txn_max)
    wallet.save()
    print(str(wallet))

def run(model_file, 
        scaler_file,
        wallet_id,
        txn_max,
        slow_sma_window
       ):
    global model
    global x_scaler
    global wallet
    global ohlcv_cache

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

    # ---  Set up currency data buffers
    # initialize ohlcv data
    ohlcv_cache = {
        "BTC":get_cryptocompare_ohlcv("BTC",slow_sma_window), 
        "ETH":get_cryptocompare_ohlcv("ETH",slow_sma_window), 
        "USDT":get_cryptocompare_ohlcv("USDT",slow_sma_window), 
        "sUSD/USDT":get_kucoin_ohlcv("sUSD/USDT",slow_sma_window)
    }

    # At this point the ohlcv data caches are preloaded with enough data to be able to generate X

    # Configure and load scheduler
    # scheduler should call on_trigger()

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sabot ready")

    # schedule.every().hour.at(":01").do(on_trigger, sma_window=slow_sma_window, txn_max=txn_max)

    for i in range(20):  # TODO: replace when scheduler is integrated
        on_trigger(slow_sma_window,txn_max)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - sabot terminating")

if __name__ == "__main__":
    fire.Fire(run)

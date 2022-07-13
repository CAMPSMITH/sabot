import boto3
import csv
import json
import requests
import time

# import pandas as pd
# from web3 import Web3
# from uniswap import Uniswap

# USDC= 6 decimals
# USDT= 6 decimals
# sUSD = 18 decimals

# Variables
bucket = "sellbucket11245"
file_name = 'sellbucketcsv.csv'

# call s3 bucket
s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket)

# lambda function
def lambda_handler(event,context):

    # Get trading data
    buyUSDTwithSUSD = requests.get('https://api.0x.org/swap/v1/quote?buyToken=USDT&sellToken=SUSD&sellAmount=20000000000000000000000')
    buyUSDTwithSUSD = buyUSDTwithSUSD.json()

    # print(buySUSDwithUSDC)
    # download s3 csv file to lambda tmp folder
    local_file_name = '/tmp/test.csv'
    bucket.download_file(file_name,local_file_name)

    total_gas_cost = float(buyUSDTwithSUSD['gasPrice'])*float(buyUSDTwithSUSD['gas'])
    gas_cost_in_USDT =float(buyUSDTwithSUSD['buyTokenToEthRate'])/(float(total_gas_cost))

    # list you want to append
    new_row = []

    new_row.append(int(time.time())) #epochtime
    new_row.append(float(buyUSDTwithSUSD['sellAmount'])/1000000000000000000) # amountofUSDCtoken
    new_row.append(buyUSDTwithSUSD['price']) #oneUSDC
    new_row.append(buyUSDTwithSUSD['guaranteedPrice']) #guaranteedPrice
    new_row.append(buyUSDTwithSUSD['protocolFee']) #protocolFee
    new_row.append(buyUSDTwithSUSD['gas']) # gas
    new_row.append(buyUSDTwithSUSD['gasPrice']) #gasprice
    new_row.append(buyUSDTwithSUSD['buyTokenToEthRate']) #buytokentoETHrate
    new_row.append(gas_cost_in_USDT)
    

    print(f'Amount of sUSD Tokens =', float(buyUSDTwithSUSD['sellAmount'])/1000000000000000000)
    print(f'One sUSD =',buyUSDTwithSUSD['price'],'USDT')
    print(f'Guaranteed Price =', buyUSDTwithSUSD['guaranteedPrice'],'USDT')
    print(f'Protocol Fee:', buyUSDTwithSUSD['protocolFee'])
    print(f'Gas:', buyUSDTwithSUSD['gas'])
    print(f'Gas Price:', buyUSDTwithSUSD['gasPrice'])
    print(f'Buy-Token to ETH Rate:', buyUSDTwithSUSD['buyTokenToEthRate'])
    print(f'Gas cost in SUSD $',gas_cost_in_USDT)
    


    # write the data into '/tmp' folder
    with open('/tmp/test.csv','r') as infile:
        list_of_csv_rows = list(csv.reader(infile))     
        list_of_csv_rows.append(new_row)

    with open('/tmp/test.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for line in list_of_csv_rows:
            writer.writerow(line)

    # upload file from tmp to s3 
    bucket.upload_file('/tmp/test.csv', file_name)

    return 


'''
rm my-deployment-package.zip
cd package
zip -r ../my-deployment-package.zip .
cd ..
zip -g my-deployment-package.zip lambda_function.py

aws lambda update-function-code \
--function-name 0xfindbesttrade \
--zip-file fileb://my-deployment-package.zip
'''

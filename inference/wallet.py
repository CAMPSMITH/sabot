import boto3
import logging
import json
import os
from datetime import datetime

# configure logger
logger = logging.getLogger()  # global handle to root logger
logger.setLevel(logging.INFO) # default log level is WARNING
if 'DEBUG' in os.environ:
    if os.environ['DEBUG'].lower() == 'true':
        # turn on debugging
        logger.setLevel(logging.DEBUG)
        logging.getLogger('boto3').setLevel(logging.DEBUG)
        logging.getLogger('botocore').setLevel(logging.DEBUG)

ddb=boto3.resource('dynamodb')
s3 = boto3.resource('s3')

# The following is a facade for the wallet
class Wallet():

  def __init__(self, wallet_id, max_trade_amount, bucket_name, log_path):
    logger.info(f"initializing wallet {wallet_id}, max_trade_amount = {max_trade_amount}")
    self.id = wallet_id
    self.max_trade_amount = max_trade_amount
    self.balances={}    
    self.log_path=log_path

    # ensure s3 bucket is valid
    assert s3 is not None, "Unable to instantiate boto3 S3 client"
    self.bucket = s3.Bucket(bucket_name)
    assert self.bucket is not None, f"Unable to get bucket {bucket_name}"

    # ensure ddb client was initialized
    assert ddb is not None, "Unable to instantiate boto3 DDB client"
    self.wallet_table = ddb.Table(os.environ['WALLET_TABLE'])
    assert self.wallet_table is not None, f"Unable to connect to table {os.environ['WALLET_TABLE']}"
    
    self.retrieve_balances()

  def retrieve_balances(self):
    response = self.wallet_table.get_item(Key={'id': self.id, 'entity': 'wallet'})
    if response is not None:
        if 'Item' in response:
            if 'balances' in response['Item']:
                self.balances=json.loads(response['Item']['balances'])
            logger.info(f"{self.id}:{'wallet'}: \n{json.dumps(self.balances)}")
        else:
            logger.info(f"no items found for {self.id}:{'wallet'}")
    else:
        logger.info(f"no items found for {self.id}:{'wallet'}")
    return self.balances

  def put_balances(self):
    self.wallet_table.put_item(
        Item={
            'id': self.id, 
            'entity': 'wallet',
            'balances': json.dumps(self.balances)}
    )

  def get_balances(self):
    return self.balances

  def get_balance(self,currency):
    if currency in self.balances:
        return self.balances[currency]['amount']
    return None

  def __str__(self):
    for balance in self.balances:
        return f"Current {balance['currency']} balance is  {balance['amount']}"

  def deposit(self,currency,amount):
    if currency in self.balances:
        self.balances[currency] = {
            "currency": currency,
            "amount":self.balances[currency]['amount'] + amount
        }   
    else:
        self.balances[currency] = {
            "currency": currency,
            "amount": amount
        }   
    self.put_balances()

  def withdraw(self,currency,amount):
    if currency in self.balances:
        self.balances[currency] = {
            "currency": currency,
            "amount":self.balances[currency]['amount'] - amount
        }
    else:
        self.balances[currency] = {
            "currency": currency,
            "amount": - amount
        }
    self.put_balances()

  def swap(self,source_currency, source_amount, target_currency):
    if source_amount > self.max_trade_amount:
        source_amount = self.max_trade_amount
    if source_amount > self.balances[source_currency]['amount']:
        source_amount = self.balances[source_currency]['amount']
    # TODO: integrate with swap transaction details
    target_amount = source_amount # FIX
    fees=0 # FIX
    self.withdraw(source_currency,source_amount)
    self.deposit(target_currency,target_amount)
    logger.info(f"swapped {source_amount} {source_currency} to {target_amount} {target_currency}. fees: {fees}")
    self.log_transaction(
        {
            "wallet_id":self.id,
            "source_currency": source_currency,
            "source_amount": source_amount,
            "target_currency": target_currency,
            "target_amount": target_amount,
            "fees": fees
        }
    )

  def log_transaction(self,transaction_record):
    now = datetime.now()
    transaction_record['time']=now.strftime("%Y-%m-%d %H:%M:%S")
    self.bucket.put_object(
        Body=json.dumps(transaction_record).encode(),
        Key=f"{self.log_path}/{self.id}/{datetime.timestamp(now)}.json"
    )
    logger.info("logging transaction")
    
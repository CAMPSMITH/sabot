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
import boto3
import logging
from wallet import Wallet

# configure logger
logger = logging.getLogger()  # global handle to root logger
logger.setLevel(logging.INFO) # default log level is WARNING
if 'DEBUG' in os.environ:
    if os.environ['DEBUG'].lower() == 'true':
        # turn on debugging
        logger.setLevel(logging.DEBUG)
        logging.getLogger('boto3').setLevel(logging.DEBUG)
        logging.getLogger('botocore').setLevel(logging.DEBUG)

wallet = None

def get_data():
    """ get data needed to get a prediction
    data needs to be scaled properly
    """
    logger.info("getting_data")
    return {}

def get_prediction(x_scaled):
    """ get data needed to get a prediction
    data needs to be scaled properly
    """
    y_predict = 1
    logger.info(f"model prediction {y_predict} for data {x_scaled}")
    return y_predict


def lambda_handler(event, context):
    """Entry point for lambda function execution.

    Args:
        event (dict): The event dictionary that is passed into the method by AWS.
        context (dict): the context dictionary that is passed into the method by AWS.

    Returns:  
        ** This function does not return any values **

    """
    try:
        # Verify config 
        assert "WALLET_ID" in os.environ, f"WALLET_ID is missing"
        logger.info(f"WALLET_ID: {os.environ['WALLET_ID']}")
        assert "MAXIMUM_TRADE_AMOUNT" in os.environ, f"MAXIMUM_TRADE_AMOUNT is missing"
        logger.info(f"MAXIMUM_TRADE_AMOUNT: {os.environ['MAXIMUM_TRADE_AMOUNT']}")
        assert "WALLET_TABLE" in os.environ, f"WALLET_TABLE is missing"
        logger.info(f"WALLET_TABLE: {os.environ['WALLET_TABLE']}")
        assert "BUCKET_NAME" in os.environ, f"BUCKET_NAME is missing"
        logger.info(f"BUCKET_NAME: {os.environ['BUCKET_NAME']}")
        assert "TRANSACTION_LOG_PATH" in os.environ, f"TRANSACTION_LOG_PATH is missing"
        logger.info(f"TRANSACTION_LOG_PATH: {os.environ['TRANSACTION_LOG_PATH']}")

        # initialize wallet
        wallet = Wallet(
            os.environ['WALLET_ID'],
            float(os.environ['MAXIMUM_TRADE_AMOUNT']),
            os.environ['BUCKET_NAME'],
            os.environ['TRANSACTION_LOG_PATH'],
        )
        assert wallet is not None, "Unable to initialize wallet"

    except AssertionError as error:
        logger.error(error)
        return
    except:
        logger.error(f"something unexpected occurred: {sys.exc_info()}")
        return

    logger.debug(f"""
    \event:
    \t=====================
    \t{event}
    \t=====================
    """)

    try:
        x_scaled = get_data()
        y_predict = get_prediction(x_scaled)
        if y_predict == 1:
            logger.info(f"trade signal is {y_predict}, buy")
            wallet.swap("USDC",wallet.get_balance("USDC"),"sUSD")
        elif y_predict == -1:
            logger.info(f"trade signal is {y_predict}, sell")
            wallet.swap("sUSD",wallet.get_balance("sUSD"),"USDC")
    except AssertionError as error:
        logger.error(error)
        return
    except:
        logger.error(f"something unexpected occurred: {sys.exc_info()}")
        return


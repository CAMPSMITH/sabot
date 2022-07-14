import json
import os
from datetime import datetime
from pathlib import Path

# The following is a Class that models a trading bot
class Wallet():

  def __init__(self, wallet_id, log_path='logs', balances = {'USDT':{'currency':'USDT','amount':100000.00}}):
    self.id = wallet_id
    self.balances={}    
    if balances is not None:
        self.balances = balances
    self.log_path=f"{log_path}/wallet-{wallet_id}.log"
    self.history_path=f"{log_path}/wallet-{wallet_id}.csv"

    if Path(self.log_path).exists():
        os.remove(self.log_path)
    if Path(self.history_path).exists():
        os.remove(self.history_path)

    self.log(f"initializing wallet {self.id}")    
    self.log(str(self))
    self.save()

  def get_balances(self):
    return self.balances

  def get_balance(self,currency):
    print(f"currency: {currency}")
    print(f"wallet: {str(self)}")
    if currency in self.balances:
      return self.balances[currency]['amount']
    return 0.00

  def __str__(self):
    result = f"wallet {self.id}:\n"
    for currency in self.balances:
        result += f"\t{currency} balance is  {self.balances[currency]['amount']}"
    return result

  def deposit(self,currency,amount):
    self.log(f"depositing {amount} {currency}")
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
    self.log(str(self))

  def withdraw(self,currency,amount):
    assert currency in self.balances, f"{currency} is not in the wallet"
    assert self.balances[currency]["amount"] >= amount, f"Insufficient funds to withdraw {amount} {currency}"
    self.log(f"wihdrawing {amount} {currency}")
    self.balances[currency] = {
        "currency": currency,
        "amount":self.balances[currency]['amount'] - amount
    }

  def save(self,header=False):
    with open(self.history_path, "a") as file_object:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        isFirst = True
        for currency in self.balances:
            if isFirst:
                isFirst = False
            else:
                line += ', '
            line += f"{self.balances[currency]['currency']}, {self.balances[currency]['amount']}"
        line += "\n"
        file_object.write(line)    
 
  def log(self,message):
    with open(self.log_path, "a") as file_object:
        file_object.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}- {message}\n")    

  def get_total_value(self):
    total = 0
    for currency in self.balances:
      total += self.balances[currency]['amount']
    return total
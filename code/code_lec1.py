#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 20:48:03 2021

@author: ruslebiffen
"""

'''
Scraping S&P 500 members from Wikipedia 
'''

import requests
from bs4 import BeautifulSoup

def get_ticks():
    # Get all SP500 tickers and industries
    URL       = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    res       = requests.get(URL).text
    soup      = BeautifulSoup(res,'lxml')
    comp_list = []
    ind_list  = []
    for items in soup.find('table', class_='wikitable').find_all('tr')[1::1]:
        row = items.find_all(['th','td'])
        try:
            comp_list.append(row[0].a.text)
            ind_list.append(row[3].text)
        except: continue
    ind_list = [ind.split('\n')[0] for ind in ind_list ]
    return comp_list, ind_list

tickers ,_ = get_ticks()





'''
Downloading stock data from Yahoo Finance

https://pypi.org/project/yfinance/#description
'''
import yfinance as yf

msft = yf.Ticker("MSFT")

# get stock info
msft.info

# Fetch data for multiple tickers: 
# Dowload price data for the first 10 companies that we scraped from Wikipedia
tickers ,_ = get_ticks()

tickers    = tickers[:10]
tickers    = ' '.join(tick for tick in tickers)

start = "2017-01-01"
end   = "2017-04-30"
data  = yf.download(tickers, start=start, end=end)

data.Close




'''
Download data from Federal Reserve Economic Data (FRED)

T-bill:
https://fred.stlouisfed.org/series/TB3MS#
'''
import pandas_datareader as web
import datetime          as dt

start     = dt.datetime(2005, 1, 1)
end       = dt.datetime(2018,1,1)
risk_free = web.get_data_fred('TB3MS',start=start,end=end)/100





























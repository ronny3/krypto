from autobahn.asyncio.wamp import ApplicationSession
from autobahn_autoreconnect import ApplicationRunner
from autobahn_autoreconnect import IReconnectStrategy

class OneSecondStrategy(IReconnectStrategy):

    def __init__(self):
        self._retry_counter = 0
  
    def get_retry_interval(self):
        """Return interval, in seconds, to wait between retries"""
        return 1 

    def reset_retry_interval(self):
        """Called before the first time we try to reconnect"""
        self._retry_counter = 0

    def increase_retry_interval(self):
        """Called every time a retry attempt fails"""
        self._retry_counter += 1
    
    def retry(self):
        """Returning True will keep retrying, False will stop retrying"""
        return self._retry_counter < 100

from asyncio import coroutine
import re
import time
from datetime import datetime
import csv
from forex_python.bitcoin import BtcConverter
import numpy as np
from asyncio import AbstractEventLoop
b = BtcConverter()
DATA = []

old_bitcoin = b.get_latest_price('USD')

#time.ctime()[0:3]
#str(datetime.now())[11:22]


# oletushinta millä btc:n saa myytyä
# tällä voi laskea muiden arvo
BTC_price = 2000
out = open('data.csv', 'a')
writer = csv.writer(out)
class PoloniexComponent(ApplicationSession):

    def onConnect(self):
        self.join(self.config.realm)
    @coroutine
    def onJoin(self, details):
        def onTicker(*args):
            x = re.split('_', args[0])
            if 'ETH' in x and 'BTC' in x:
                print("Ticker event received:", x, "last:", args[1], "lowestAsk:", args[2], "highestBid:", args[3], "percentChange:", args[4], "baseVol:", args[5], "quoteVol:", args[6])
                      #,"isFrozen:", args[7], "24hrHigh:", args[8], "24hrlow:", args[9])
                BTC_price = b.get_latest_price('USD')-50
                print("ETH price:", float(args[1])*BTC_price, time.ctime()[0:3], str(datetime.now())[11:22])
                with open('data.csv', 'a') as out:
                    writer = csv.writer(out)
                    writer.writerow((float(args[1])*BTC_price, time.ctime()[0:3], str(datetime.now())[11:22]))
        
        def trade(*args, **kwargs):
                for i in range(len(args)):
                    try:   
                        rivi = args[i]["data"] #date, total, type, rate, amount, tradeID
#                            print(rivi["date"][11:19], rivi["type"], rivi["rate"], rivi["amount"])
                        tunnit = (int(rivi["date"][11:13])-12)/12
                        if rivi["type"] == "sell":
                            event = 0
                        else:
                            event = 1
                        rate = np.array(rivi["rate"]).astype('float32')
                        amount = np.array(rivi["amount"]).astype('float32')
                        bitcoin = None
                        bitcoin = b.get_latest_price('USD')
                        if bitcoin == None:
                            bitcoin = old_bitcoin
                        old_bitcoin=bitcoin
                        bitcoin = bitcoin-50
                        writer.writerow((tunnit, rate, amount, event, bitcoin, rivi["date"][11:19]))
#                        DATA.append(tunnit+","+amount+","+event)
                        print (rivi)
                    except IndexError:
                        1
            #                print("XD", args)
                    except KeyError:
                        2
#                print("cF", args)
        try:
#            AbstractEventLoop.call_soon()y self.subscribe(trade, 'BTC_ETH')
            yield from self.subscribe(trade, 'BTC_ETH')
        except Exception as e:
            print("Could not subscribe to topic:", e)


def main():
    try:
        runner = ApplicationRunner("wss://api.poloniex.com:443", "realm1", retry_strategy=OneSecondStrategy(), auto_ping_timeout=2700)
        runner.run(PoloniexComponent)
    except:
        print("asdasdsad")
        


if __name__ == "__main__":
    main()

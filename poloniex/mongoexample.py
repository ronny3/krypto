#!/usr/bin/python
# -*- coding: utf-8 -*-
# core
from multiprocessing import Process
# pip
from pymongo import MongoClient
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.wamp import ApplicationSession#, ApplicationRunner
import json
from urllib.request import urlopen
from tkinter import messagebox
import time
#%%

import six
import inspect
import binascii
from functools import reduce

import txaio
txaio.use_twisted()  # noqa

from twisted.internet.defer import inlineCallbacks, succeed

from autobahn.util import public

from autobahn.websocket.util import parse_url as parse_ws_url
from autobahn.rawsocket.util import parse_url as parse_rs_url

from autobahn.twisted.websocket import WampWebSocketClientFactory
from autobahn.twisted.rawsocket import WampRawSocketClientFactory

from autobahn.websocket.compress import PerMessageDeflateOffer, \
    PerMessageDeflateResponse, PerMessageDeflateResponseAccept

from autobahn.wamp import protocol, auth
from autobahn.wamp.interfaces import IAuthenticator
from autobahn.wamp.types import ComponentConfig


__all__ = [
    'ApplicationSession',
    'ApplicationSessionFactory',
    'ApplicationRunner',
    'Application',
    'Service',

    # new API
    'Session',
    # 'run',  # should probably move this method to here? instead of component
]

try:
    from twisted.application import service
except (ImportError, SyntaxError):
    # Not on PY3 yet
    service = None
    __all__.pop(__all__.index('Service'))
    
    
class ApplicationRunner(object):
    """
    This class is a convenience tool mainly for development and quick hosting
    of WAMP application components.
    It can host a WAMP application component in a WAMP-over-WebSocket client
    connecting to a WAMP router.
    """

    log = txaio.make_logger()

    def __init__(self,
                 url,
                 realm=None,
                 extra=None,
                 serializers=None,
                 ssl=None,
                 proxy=None,
                 headers=None):
        """
        :param url: The WebSocket URL of the WAMP router to connect to (e.g. `ws://somehost.com:8090/somepath`)
        :type url: str
        :param realm: The WAMP realm to join the application session to.
        :type realm: str
        :param extra: Optional extra configuration to forward to the application component.
        :type extra: dict
        :param serializers: A list of WAMP serializers to use (or None for default serializers).
           Serializers must implement :class:`autobahn.wamp.interfaces.ISerializer`.
        :type serializers: list
        :param ssl: (Optional). If specified this should be an
            instance suitable to pass as ``sslContextFactory`` to
            :class:`twisted.internet.endpoints.SSL4ClientEndpoint`` such
            as :class:`twisted.internet.ssl.CertificateOptions`. Leaving
            it as ``None`` will use the result of calling Twisted's
            :meth:`twisted.internet.ssl.platformTrust` which tries to use
            your distribution's CA certificates.
        :type ssl: :class:`twisted.internet.ssl.CertificateOptions`
        :param proxy: Explicit proxy server to use; a dict with ``host`` and ``port`` keys
        :type proxy: dict or None
        :param headers: Additional headers to send (only applies to WAMP-over-WebSocket).
        :type headers: dict
        """
        assert(type(url) == six.text_type)
        assert(realm is None or type(realm) == six.text_type)
        assert(extra is None or type(extra) == dict)
        assert(headers is None or type(headers) == dict)
        assert(proxy is None or type(proxy) == dict)
        self.url = url
        self.realm = realm
        self.extra = extra or dict()
        self.serializers = serializers
        self.ssl = ssl
        self.proxy = proxy
        self.headers = headers

        # this if for auto-reconnection when Twisted ClientService is avail
        self._client_service = None
        # total number of successful connections
        self._connect_successes = 0

#    def stop(self):
#        """
#        Stop reconnecting, if auto-reconnecting was enabled.
#        """
#        self.log.debug('{klass}.stop()', klass=self.__class__.__name__)
#        if self._client_service:
#            return self._client_service.stopService()
#        else:
#            return succeed(None)

    def run(self, make, start_reactor=True, auto_reconnect=True, log_level='info'):
        """
        Run the application component.
        :param make: A factory that produces instances of :class:`autobahn.twisted.wamp.ApplicationSession`
           when called with an instance of :class:`autobahn.wamp.types.ComponentConfig`.
        :type make: callable
        :param start_reactor: When ``True`` (the default) this method starts
           the Twisted reactor and doesn't return until the reactor
           stops. If there are any problems starting the reactor or
           connect()-ing, we stop the reactor and raise the exception
           back to the caller.
        :returns: None is returned, unless you specify
            ``start_reactor=False`` in which case the Deferred that
            connect() returns is returned; this will callback() with
            an IProtocol instance, which will actually be an instance
            of :class:`WampWebSocketClientProtocol`
        """
        if start_reactor:
            # only select framework, set loop and start logging when we are asked
            # start the reactor - otherwise we are running in a program that likely
            # already tool care of all this.
            from twisted.internet import reactor
            txaio.use_twisted()
            txaio.config.loop = reactor
            txaio.start_logging(level=log_level)

        if callable(make):
            # factory for use ApplicationSession
            def create():
                cfg = ComponentConfig(self.realm, self.extra)
                try:
                    session = make(cfg)
                except Exception:
                    self.log.failure('ApplicationSession could not be instantiated: {log_failure.value}')
                    if start_reactor and reactor.running:
                        reactor.stop()
                    raise
                else:
                    return session
        else:
            create = make

        if self.url.startswith(u'rs'):
            # try to parse RawSocket URL ..
            isSecure, host, port = parse_rs_url(self.url)

            # use the first configured serializer if any (which means, auto-choose "best")
            serializer = self.serializers[0] if self.serializers else None

            # create a WAMP-over-RawSocket transport client factory
            transport_factory = WampRawSocketClientFactory(create, serializer=serializer)

        else:
            # try to parse WebSocket URL ..
            isSecure, host, port, resource, path, params = parse_ws_url(self.url)

            # create a WAMP-over-WebSocket transport client factory
            transport_factory = WampWebSocketClientFactory(create, url=self.url, serializers=self.serializers, proxy=self.proxy, headers=self.headers)

            # client WebSocket settings - similar to:
            # - http://crossbar.io/docs/WebSocket-Compression/#production-settings
            # - http://crossbar.io/docs/WebSocket-Options/#production-settings

            # The permessage-deflate extensions offered to the server ..
            offers = [PerMessageDeflateOffer()]

            # Function to accept permessage_delate responses from the server ..
            def accept(response):
                if isinstance(response, PerMessageDeflateResponse):
                    return PerMessageDeflateResponseAccept(response)

            # set WebSocket options for all client connections
            transport_factory.setProtocolOptions(maxFramePayloadSize=1048576,
                                                 maxMessagePayloadSize=1048576,
                                                 autoFragmentSize=65536,
                                                 failByDrop=False,
                                                 openHandshakeTimeout=55,
                                                 closeHandshakeTimeout=40.,
                                                 tcpNoDelay=True,
                                                 autoPingInterval=1.,
                                                 autoPingTimeout=60,
                                                 autoPingSize=4,
                                                 perMessageCompressionOffers=offers,
                                                 perMessageCompressionAccept=accept)

        # supress pointless log noise
        transport_factory.noisy = False

        # if user passed ssl= but isn't using isSecure, we'll never
        # use the ssl argument which makes no sense.
        context_factory = None
        if self.ssl is not None:
            if not isSecure:
                raise RuntimeError(
                    'ssl= argument value passed to %s conflicts with the "ws:" '
                    'prefix of the url argument. Did you mean to use "wss:"?' %
                    self.__class__.__name__)
            context_factory = self.ssl
        elif isSecure:
            from twisted.internet.ssl import optionsForClientTLS
            context_factory = optionsForClientTLS(host)

        from twisted.internet import reactor
        if self.proxy is not None:
            from twisted.internet.endpoints import TCP4ClientEndpoint
            client = TCP4ClientEndpoint(reactor, self.proxy['host'], self.proxy['port'])
            transport_factory.contextFactory = context_factory
        elif isSecure:
            from twisted.internet.endpoints import SSL4ClientEndpoint
            assert context_factory is not None
            client = SSL4ClientEndpoint(reactor, host, port, context_factory)
        else:
            from twisted.internet.endpoints import TCP4ClientEndpoint
            client = TCP4ClientEndpoint(reactor, host, port)

        # as the reactor shuts down, we wish to wait until we've sent
        # out our "Goodbye" message; leave() returns a Deferred that
        # fires when the transport gets to STATE_CLOSED
        def cleanup(proto):
            if hasattr(proto, '_session') and proto._session is not None:
                if proto._session.is_attached():
                    return proto._session.leave()
                elif proto._session.is_connected():
                    return proto._session.disconnect()

        # when our proto was created and connected, make sure it's cleaned
        # up properly later on when the reactor shuts down for whatever reason
        def init_proto(proto):
            self._connect_successes += 1
            reactor.addSystemEventTrigger('before', 'shutdown', cleanup, proto)
            return proto

        use_service = False
        if auto_reconnect:
            try:
                # since Twisted 16.1.0
                from twisted.application.internet import ClientService
                from twisted.application.internet import backoffPolicy
                use_service = True
            except ImportError:
                use_service = False

        if use_service:
            # this code path is automatically reconnecting ..
            self.log.debug('using t.a.i.ClientService')

            default_retry = backoffPolicy()

            if False:
                # retry policy that will only try to reconnect if we connected
                # successfully at least once before (so it fails on host unreachable etc ..)
                def retry(failed_attempts):
                    print("xddd")
                    if self._connect_successes > 0:
                        return default_retry(failed_attempts)
                    else:
                        self.stop()
                        return 100000000000000
            else:
                retry = default_retry

            self._client_service = ClientService(client, transport_factory, retryPolicy=retry)
            self._client_service.startService()

            d = self._client_service.whenConnected()

        else:
            # this code path is only connecting once!
            self.log.debug('using t.i.e.connect()')

            d = client.connect(transport_factory)

        # if we connect successfully, the arg is a WampWebSocketClientProtocol
        d.addCallback(init_proto)

        # if the user didn't ask us to start the reactor, then they
        # get to deal with any connect errors themselves.
        if start_reactor:
            # if an error happens in the connect(), we save the underlying
            # exception so that after the event-loop exits we can re-raise
            # it to the caller.
            print("xdd")
            class ErrorCollector(object):
                exception = None

                def __call__(self, failure):
                    self.exception = failure.value
                    reactor.stop()
            connect_error = ErrorCollector()
            d.addErrback(connect_error)

            # now enter the Twisted reactor loop
            reactor.run()

            # if we exited due to a connection error, raise that to the
            # caller
            if connect_error.exception:
                print("xd")
                raise connect_error.exception

        else:
            # let the caller handle any errors
            return d
#%%
# git
from poloniex import Poloniex
import numpy as np
from forex_python.bitcoin import BtcConverter
b = BtcConverter()
old_bitcoin = None
while (old_bitcoin == None):
    old_bitcoin = b.get_latest_price('USD')
bitcoin = old_bitcoin
from time import gmtime, strftime

class WAMPTicker(ApplicationSession):
    """ WAMP application - subscribes to the 'ticker' push api and saves pushed
    data into a mongodb """
    @inlineCallbacks
    def onJoin(self, details):
        self.__time = 0
        self.__bitcoin = bitcoin
        self.db = MongoClient().poloniex['Cryptos']
        self.db.drop()
        yield self.subscribe(self.onETH, 'BTC_ETH') # ethereum
        yield self.subscribe(self.onXRP, 'BTC_XRP') # ripple
        yield self.subscribe(self.onETC, 'BTC_ETC') # eth classic
        yield self.subscribe(self.onLTC, 'BTC_LTC') # eth classic
        print('Subscribed to Ticker')
        
    def onETH(self, *data, **seq):
        if (time.time()-self.__time>2):            
            self.__bitcoin = float(json.loads(urlopen('https://www.bitstamp.net/api/ticker/').read().decode('utf8'))['last'])
            print("BTC usd:",self.__bitcoin)
            self.__time = time.time()

        for i in range(len(data)):
            try:
                rivi = data[i]["data"] #date, total, type, rate, amount, tradeID
                tunnit = (int(rivi["date"][11:13])-12)/12
                if rivi["type"] == "sell":
                    event = 0
                else:
                    event = 1
                rate = rivi["rate"]
                amount =rivi["amount"]
                self.db.ETH.insert_one(
                    {
                        "_id": rivi["tradeID"],
                        "rate": rate,
                        "amount": amount,
                        "bitcoin": self.__bitcoin,
                        "klo":rivi["date"][11:19],
                        "tunnit":tunnit,
                        "event": event
                        })
                print ("ETH:",rivi)
            except:
                1
    
    def onXRP(self, *data, **seq):
        for i in range(len(data)):
            try:   
                rivi = data[i]["data"] #date, total, type, rate, amount, tradeID
                tunnit = (int(rivi["date"][11:13])-12)/12
                if rivi["type"] == "sell":
                    event = 0
                else:
                    event = 1
                rate = rivi["rate"]
                amount =rivi["amount"]
                self.db.XRP.insert_one(
                    {
                        "_id": rivi["tradeID"],
                        "rate": rate,
                        "amount": amount,
                        "bitcoin": self.__bitcoin,
                        "klo":rivi["date"][11:19],
                        "tunnit":tunnit,
                        "event": event
                        })
                print ("XRP:",rivi)
            except:
                1
          
    def onETC(self, *data, **seq):
        for i in range(len(data)):
            try:   
                rivi = data[i]["data"] #date, total, type, rate, amount, tradeID
                tunnit = (int(rivi["date"][11:13])-12)/12
                if rivi["type"] == "sell":
                    event = 0
                else:
                    event = 1
                rate = rivi["rate"]
                amount =rivi["amount"]
                self.db.ETC.insert_one(
                    {
                        "_id": rivi["tradeID"],
                        "rate": rate,
                        "amount": amount,
                        "bitcoin": self.__bitcoin,
                        "klo":rivi["date"][11:19],
                        "tunnit":tunnit,
                        "event": event
                        })
                print ("ETC:",rivi)
            except:
                1
                          
    def onLTC(self, *data, **seq):
        for i in range(len(data)):
            try:   
                rivi = data[i]["data"] #date, total, type, rate, amount, tradeID
                tunnit = (int(rivi["date"][11:13])-12)/12
                if rivi["type"] == "sell":
                    event = 0
                else:
                    event = 1
                rate = rivi["rate"]
                amount =rivi["amount"]
                self.db.LTC.insert_one(
                    {
                        "_id": rivi["tradeID"],
                        "rate": rate,
                        "amount": amount,
                        "bitcoin": self.__bitcoin,
                        "klo":rivi["date"][11:19],
                        "tunnit":tunnit,
                        "event": event
                        })
                print ("LTC: ",rivi)
            except:
                1
                
    def onDisconnect(self):
        # stop reactor if disconnected
        if reactor.running:
            reactor.stop()


class Ticker(object):

    def __init__(self):
        self.running = False
        # open/create poloniex database, ticker collection/table
        self.db = MongoClient().poloniex['ticker']
        # thread namespace
        self._appProcess = None
        self._appRunner = ApplicationRunner(
            u"wss://api.poloniex.com:443", u"realm1"
        )

    def __call__(self, market='USDT_BTC'):
        """ returns ticker from mongodb """
        return self.db.getCollection('BTC_ETH.ETH_BTC').count()

    def start(self):
        """ Start WAMP application runner process """
        self._appProcess = Process(
            target=self._appRunner.run(make=WAMPTicker)
        )
        self._appProcess.daemon = True
        self._appProcess.start()
        self.running = True

    def stop(self):
        """ Stop WAMP application """
        try:
            self._appProcess.terminate()
        except:
            pass
        try:
            self._appProcess.join()
        except:
            pass
        self.running = False

if __name__ == '__main__':
    from time import sleep
    ticker = Ticker()
    ticker.start()
#    while(1):
##        print(ticker())
#        sleep(1000)
#    ticker()
#    sleep(1000)
#    for i in range(100):
#        sleep(10)
#        print("USDT_BTC: lowestAsk= %s" % ticker()['lowestAsk'])
    ticker.stop()
    print("Done")

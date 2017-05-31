#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
class Plots():
    def __init__():
        1
    
    def WindowPlot(X_train, y_train, indeksi=None):
        """ Plots random window from windowed data unless index is given,
            then plot index
        """
        from random import randint
        from matplotlib.pyplot import subplots, title, show
        if not indeksi:
            indeksi = randint(0, len(X_train)-1)
        price = X_train[indeksi, :, 0]
        volume = X_train[indeksi, :, 1]
        x = np.linspace(1, len(price), len(price))
        target = y_train[indeksi]
        
        fig, ax1 = subplots()
        ax1.plot(x, price, 'b-')
        ax1.set_xlabel('tick')
        ax1.set_ylabel('price', color='b')
        ax1.tick__params('y', colors='b')
        ax2 = ax1.twinx()
        ax2.plot(x, volume, 'r.')
        ax2.set_ylabel('volume', color='r')
        ax2.tick_params('y', colors='r')
        
        fig.tight_layout()
        title(target)
        show()
        
    def LongDataPlot(data):
        
        """ Plots long segment of data"""
        
        
        


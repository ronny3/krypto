#!/usr/bin/env python3
# -*- coding: utf-8
import numpy as np
from pymongo import MongoClient
import pymongo

class DBreader():
    def __init__(self):
        self.__features = []
        self.__times = []
        self.__ids = []
    def ReadDoc(self, document, include_id = False, include_time = False):
        """
        # Returns
            time, iid, (1,2)-feature array
        """
        iid = int(document["_id"])
        amount = np.array(document["amount"], dtype='float64')
        bitcoin = np.array(document["bitcoin"], dtype='float64')
        time = document["klo"]
        rate = np.array(document["rate"], dtype='float64')
        value = bitcoin*rate # = coin-to-usd
        if include_id:
            if include_time:
                return time, iid, np.array((value, amount))
            return iid, np.array((value, amount))
        if include_time:
            return time, np.array((value, amount))
        return np.array((value, amount))
        
    def SetCollectionName(self, collection_name):
        db = MongoClient().poloniex
        self.__collection = db[collection_name]
        
    def readCompleteDatabase(self):
        """ Reads the database into lists"""
        print("Reading",self.__collection.count(),"files")
        files = self.__collection.find({})
        for cursor in files:
            time, iid, features = self.read_doc(cursor, True, True)
            self.__features.append(features)
            self.__times.append(time)
            self.__ids.append(iid)
        
    def WindowData(self, window_size, wait_time):
        data = []
        feature_vec = []
        prev_id = self.__collection.find_one({})["_id"]
        for i in range(0, len(self.__ids)):
            if (self.__ids[i] != prev_id):
                prev_id = self.__ids[i]
                data.append(feature_vec)
                feature_vec = []
                feature_vec.append(self.__features[i])
            else:
                prev_id = self.__ids[i]
                feature_vec.append(self.__features[i])
                
        cut_data = []
        for seq in data:
            if len(seq)>window_size + wait_time:
                cut_data.append(seq)
        data = [] # release memory
        features = np.zeros((window_size, 2))
        targets = []
        feature_vec = []
        feature_world = []
        target_world = []
        
        from math import floor
        for data in cut_data:
            try:
                added = 0
                i = 0
                while(i<len(data)):
                    seq=data[i]
                    if added == window_size:
                        added = 0
                        target = data[i-1+wait_time-3:i-1+wait_time+3]# avg 
                        value = target[0] 
                        value = np.average(value)
                        feature_vec.append(features)
                        targets.append(value)
                        features = np.zeros((window_size, 2))
                        i -= (window_size-50) #sliding window
                        
                    features[added,:] = seq
                    added += 1
                    i += 1
                    
                feature_world.append(np.array(feature_vec))
                target_world.append(np.array(targets))
                feature_vec=[]
                targets=[]
            
            except IndexError:
                feature_world.append(np.array(feature_vec))
                target_world.append(np.array(targets))
                feature_vec = []
                targets= []
                
        X_train = feature_world[0]
        y_train = target_world[0]
                    
        
        
    
        
    
    
    
    
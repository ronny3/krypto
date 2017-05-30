from numpy import genfromtxt
import numpy as np
from pymongo import MongoClient
import pymongo

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def read_doc(dokumentti):
    iid = int(dokumentti["_id"]) #id
    amount = dokumentti["amount"] # volume
    amount = np.array(amount, dtype='float64')
    bitcoin = dokumentti["bitcoin"] # btc/usd
    bitcoin = np.array(bitcoin, dtype='float64')
#    event = dokumentti["event"] # sell/buy
    klo = dokumentti["klo"] # tapahtuman aika
    rate = dokumentti["rate"] # vaihtorate, tärkee
    rate = np.array(rate, dtype='float64')
#    tunnit = dokumentti["tunnit"] # [-1,1] = 0-24
    value = (bitcoin*rate)
    features = np.array((value, amount))
    return klo, iid, features

def read_database(collection_name, ikkunan_pituus, odotusaika, to_classes=False, split=False):
    db = MongoClient().poloniex
    collection = db[collection_name]
    collection.count()
    eka = collection.find_one({})
    edellinen_id = eka["_id"]
    kaikki = collection.find({})
    data = []
    featuret = []
    for cursor in kaikki:
        klo, iid, features = read_doc(cursor)
        if (iid-1 != edellinen_id):
            edellinen_id=iid
            data.append(featuret)
            featuret=[]
            featuret.append(features)
        else:
            edellinen_id=iid
            featuret.append(features)
    real_data=[]
    
    for seq in data:
        if len(seq)>ikkunan_pituus+odotusaika:
            real_data.append(seq)
    features = np.zeros((ikkunan_pituus,2))
    featuret=[]
    targets=[]
    feature_world=[]
    target_world=[]
    
    from math import floor
    for a in real_data:
        try:
            added=0
            i=0
            while i<len(a):
                seq=a[i]
                if added == ikkunan_pituus:
                    added = 0
                    target = a[i-1+odotusaika]
                    value = target[0]
#                    value = value - features[ikkunan_pituus-1,0]
                    featuret.append(features)
                    targets.append(value)
                    features = np.zeros((ikkunan_pituus,2))
                    i -= floor(ikkunan_pituus/1.1)
                    
                features[added,:] = seq
                added += 1
                i += 1
            train_data = np.array(featuret)
            feature_world.append(train_data)
            target_world.append(np.array(targets))
            featuret=[]
            targets=[]
        except IndexError:
            train_data = np.array(featuret)
            feature_world.append(train_data)
            target_world.append(np.array(targets))
            featuret=[]
            targets=[]
            
    X_train=feature_world[0]
    y_train=target_world[0]
    for i in range(len(feature_world)-1):
        i=i+1
        X_train = np.concatenate((X_train,feature_world[i]),axis=0)
        y_train = np.concatenate((y_train,target_world[i]),axis=0)
    if to_classes:
        loppuRivi = X_train[:,X_train.shape[1]-1,0]
        y_train=y_train-loppuRivi
        for i in range (0,len(y_train)):
            if (y_train[i] > (-0.15) and y_train[i] < (0.15)):
                y_train[i] = 2
            elif y_train[i]<-0.15 and y_train[i]>-1:
                y_train[i] = 1
            elif y_train[i]<-1:
                y_train[i] = 0
            elif y_train[i]>0.15 and y_train[i]<1:
                y_train[i] = 3
            elif y_train[i]>1:
                y_train[i] = 4
    if split:
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split, stratify=y_train)
        except:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split)
        return X_train, y_train, X_val, y_val
    return X_train, y_train

from random import randint
import matplotlib.pyplot as plt
def plotti(X_train,y_train, indeksi=None):
    if indeksi==None:
        indeksi = randint(0,len(X_train)-1)
    price = X_train[indeksi,:,0]
    volume = X_train[indeksi,:,1]
    x = np.linspace(1, len(price), len(price))
    target = y_train[indeksi]
    
    fig, ax1 = plt.subplots()
    ax1.plot(x,price, 'b-')
    ax1.set_xlabel('tick')
    ax1.set_ylabel('price', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(x, volume, 'r.')
    ax2.set_ylabel('volume', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    plt.title(target)
    plt.show()
def normalize(X_train):
    for i in range(len(X_train)):
        X_train[i,:,0] -= np.mean(X_train[i,:,0])
        X_train[i,:,1] -= np.mean(X_train[i,:,1])
#        X_train[i,:,0] /= np.std(X_train[i,:,0])
#        X_train[i,:,1] /= np.std(X_train[i,:,1])
    return X_train


def moving_average(a, n=5) :
    if n==1:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
def move(X_train, pituus, n):
    X = np.zeros((len(X_train),pituus-n+1,2))
    for i in range(len(X_train)):
        X[i,:,0] = moving_average(X_train[i,:,0], n=n)
        X[i,:,1] = moving_average(X_train[i,:,1], n=n)
    return X
    
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def modeli(search_params):
    from sklearn.ensemble import RandomTreesEmbedding
    from scipy.sparse import csr_matrix
    X_val = X_train[-1800:,:,:]
    X_train = X_train[0:15000,:,:]
    y_val = y_train[-1800:]
    y_train = y_train[0:15000]
    clf = RandomTreesEmbedding(n_estimators=1)
    clf = clf.fit(X_train[:,:,0],y_train)
    g=clf.transform(X_val[:,:,0])
    f=g.todense()
    q=clf.predict(X_val[:,:,0])
    a=0
    d=0
    f=0
    for i in range(q.shape[0]):
        if q[i]==y_val[i]:
            a +=1
            if q[i]==4:
                d +=1
            if (q[i]==0):
                f += 1
        else:
            1
    b = a/q.shape[0]
    c = np.count_nonzero(q==4)
    g = np.count_nonzero(q==0)
    h = f/g
    e=d/c
    
    ikkunan_pituus = randint(search_params["pituus_vali"][0],search_params["pituus_vali"][1])
    odotusaika = randint(search_params["odotusaika_vali"][0], search_params["odotusaika_vali"][1])
    avg = randint(search_params["avg_vali"][0], search_params["avg_vali"][1])
    crypto = search_params["coini"][randint(0,3)]
    optimizer = search_params["optimizer"][randint(0,3)]
    loss = search_params["loss"][randint(0,1)]
    neuronien_lkm = randint(search_params["neuronien_vali"][0], search_params["neuronien_vali"][1])
    mapsit = randint(search_params["conv_vali"][0], search_params["conv_vali"][1])
    stridet = randint(search_params["stridet"][0], search_params["stridet"][1])
    
    avg=1
    crypto='ETH'
    n_epochs=5
    ikkunan_pituus=60
    odotusaika=25
    
    X_train, y_train= read_database('Cryptos.'+crypto, ikkunan_pituus, odotusaika, to_classes=True,split=0)
    sns.countplot(y_train)
    normalize(X_train)
    normalize(X_val)
#    X_train = move(X_train, pituus=ikkunan_pituus, n=avg)
#    X_val = move(X_val, pituus=ikkunan_pituus, n=avg)

    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers.convolutional import Conv1D
    from keras.layers.local import LocallyConnected1D
    from keras.layers import Dense, Input, Flatten, Dropout, Activation, LSTM
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.callbacks import EarlyStopping
    from keras.layers.pooling import GlobalMaxPooling1D
    from keras.backend import clear_session
    from imblearn.over_sampling import RandomOverSampler
    from keras.optimizers import SGD as sgd

    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(X_train)
    inputti = (ikkunan_pituus-avg+1)
    #    return_sequences=False
#    if LSTM_lkm>1:
#        return_sequences=True
    model = Sequential()
#        model.add(LSTM(units=mapsit, activation = 'tanh',recurrent_dropout=0.75, dropout=0.5,implementation=2, return_sequences=return_sequences))
#        model.add(LSTM(units=mapsit, activation = 'tanh', recurrent_dropout=0.75, dropout=0.5))
#    model.add(BatchNormalization())
    model.add(LSTM(units=5, batch_input_shape = (1,inputti,2),implementation=2, stateful=True))
#    model.add(Dropout(0.5))
##    model.add(Flatten())
    model.add(Dense(neuronien_lkm, activation='elu',input_shape=(152,)))
#    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax'))
    model.compile(optimizer='rmsprop', loss = 'mse', metrics=[])
    for i in range(n_epochs):   
        model.fit(X_train, y_train,verbose=1, batch_size=1, epochs=1)#, validation_data = (X_val, y_val),
                      #callbacks = [EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=200)])
        model.reset_states()
    a=model.predict(X_og, batch_size=1)
    clear_session()
    return [ikkunan_pituus, odotusaika, avg, neuronien_lkm, mapsit, stridet, np.std(preds)]

#%%
#ikkunan_pituus = 400
#odotusaika = 25
#avg=25
#X_train, y_train, X_val, y_val = read_database('Cryptos.ETH', ikkunan_pituus, odotusaika, to_classes=False,split=0.2)
#normalize(X_train)
#normalize(X_val)
#X_train = move(X_train, pituus=ikkunan_pituus, n=avg)
#X_val = move(X_val, pituus=ikkunan_pituus, n=avg)
##plotti(X_train,y_train)


## model(conv_layers=x, dense_layers=y)
#model = Sequential([
#    LocallyConnected1D(500,2, activation = ELU(alpha=1.0), input_shape = (2,ikkunan_pituus-avg+1)),
#    Dropout(0.5),   
#    Flatten(),
#    Dense(32),
#    Dropout(0.5),      
#    Activation('relu'),
#    Dense(1)
#])
#    
#model.compile(optimizer='rmsprop', loss = 'mean_absolute_error', metrics=[])
#[0.27455040185075058] [176, 25, 43, 86, 51]
from keras.layers.advanced_activations import ELU
search_params ={
        "activation":["relu", ELU(), "sigmoid"],
        "loss": ["MSE", "MAE"],
        "optimizer": ["sgd", "adadelta", "rmsprop", "adam"],
        "pituus_vali": [400, 704],
        "odotusaika_vali": [50,50],
        "avg_vali": [1, 1],
        "coini": ["ETH", "ETC", "XRP", "LTC"],
        "neuronien_vali":[35, 35],
        "conv_vali":[35,35],
        "stridet":[3,10]
        }
tiedot =[]
import time
aika = time.time()+(3600*(1/60)*75) #1234
while(time.time()<aika):
    try:
        a=modeli(search_params)
        print(a[0], a[1])
        tiedot.append(a)
    except:
        print("error")
        1
#%%

loss = []
muut0 = []
muut1 = []
muut2 = []
muut3 = []
muut4 = []
muut5 = []
muut6 = []
muut7 = []

def normalala(muut):
    maxi = max(muut)
    for i in range (0, len(muut)):
        muut[i] /= maxi
    return muut
for values in tiedot: #ikkunan_pituus 0,odotusaika 1,avg 2,neuronit 3,mapsit 4,lstmt 5, epokit 6, std7
    loss.append(values[0])
    muut0.append(values[1][0])
    muut1.append(values[1][1])
    muut2.append(values[1][2])
    muut3.append(values[1][3])
    muut4.append(values[1][4])
    muut5.append(values[1][5])
    muut6.append(values[1][6])
    muut7.append(values[1][7])

#muut0 = normalala(muut0)
#muut1 = normalala(muut1)
#muut2 = normalala(muut2)
#muut3 = normalala(muut3)
#muut4 = normalala(muut4)
#muut5 = normalala(muut5)
#muut6 = normalala(muut6)
#muut7 = normalala(muut7)


import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111)

#ax1.scatter(muut0, loss, label='pituus')
#ax1.scatter(muut1 ,loss, label='odotus')
#ax1.scatter(muut2 ,loss, label='avg')
#ax1.scatter(muut3 ,loss, label='neuronit')
#ax1.scatter(muut4 ,loss, label='mapsit')
#ax1.scatter(muut5 ,loss, label='lstmt')
#ax1.scatter(muut6 ,loss, label='epokit')
#ax1.scatter(muut7 ,loss, label='std')

plt.legend(loc='upper left');
plt.show()

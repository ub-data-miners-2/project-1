# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
# import random

import numpy as np
import pandas as pd
from numpy import genfromtxt

def load_data():
    # train_data1 = genfromtxt('2005_data-lite.csv', delimiter=',',  skip_header=1, dtype=str)
    # train_data2 = genfromtxt('2006_data-lite.csv', delimiter=',', skip_header=1, dtype=str)
    # train_data3 = genfromtxt('2007_data-lite.csv', delimiter=',', skip_header=1, dtype=str)
    # train_data = np.concatenate((train_data1, train_data2, train_data3), axis=0)
    # test_data  = genfromtxt('2008_data-lite.csv', delimiter=',',  skip_header=1, dtype=str)
    train_data = genfromtxt('data_processed.csv', delimiter=',', skip_header=1, dtype=str)

    train_data = pd.DataFrame(data=train_data, columns=train_data[0, :], dtype=str)
    # test_data  = pd.DataFrame(data=test_data, columns=test_data[0, :], dtype=str)
    #
    # # cleanup the data by removing rows with missing values
    train_data.replace('', np.nan, inplace=True)
    train_data.dropna(axis=0, how='any', inplace=True)
    test_data.replace('', np.nan, inplace=True)
    test_data.dropna(axis=0, how='any', inplace=True)

    # shuffle the data
    train_data = shuffle(train_data[1:].values)
    test_data = shuffle(test_data[1:].values)


    # train_data = shuffle(train_data)
    # test_data = shuffle(test_data)
    #
    # dict = {}
    # trainY = train_data[:,0]
    # trainX = preprocessing(train_data[:,1:], dict)
    # testY = test_data[:,0]
    # testX  = preprocessing(test_data[:,1:], dict)

    # testY = test_data.iloc[:,0]
    # testX = test_data.iloc[:,1:]
    # trainY = train_data.iloc[:,0]
    # trainX = train_data.iloc[:,1:]

    (trainX, testX, trainY, testY) = train_test_split(trainX,trainY, test_size=0.25, random_state=42)

    # Uncomment to use PCA first, and change the shape number in line 64
    # pca = PCA(n_components=11)
    # trainX = pca.fit_transform(trainX)
    # testX  = pca.fit_transform(testX)
    return [trainX, testX, trainY, testY]

# This changes the alpha numeric field into a unique number
def preprocessing(data, dict = {}):
    count = 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            d = data[i][j]
            # print("blah blah " + str(d))
            if not d.isnumeric():
                # print("hello mama")
                if not dict.get(d):
                    # print("changing " + str(data[i][j]) + " into " + str(count) )
                    dict[d] = count
                    count += 1

                data[i][j] = dict[d]
            else:
                data[i][j] = float(d)

    return data

(trainX, testX, trainY, testY) = load_data()
# encode the labels as 1-hot vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# # define the 27-220-12 architecture using Keras
model = Sequential()
model.add(Dense(100, input_shape=(trainX.shape[1],), activation="sigmoid"))
model.add(Dense(11, activation="softmax"))

# train the model using SGDprint("[INFO] training network...")
# opt = SGD(lr=0.15, momentum=0.9)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=10, batch_size=500)

# evaluate the networkprint("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=5000)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1)))

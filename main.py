# import the necessary packages
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

# from sklearn.datasets import load_iris
import numpy as np
from numpy import genfromtxt

def load_data():
    train_data = genfromtxt('2005_data.csv', delimiter=',', skip_header=1, dtype=str)
    test_data  = genfromtxt('2006_data.csv', delimiter=',', skip_header=1, dtype=str)

    dict = {}
    train_data = preprocessing(train_data, dict)
    test_data  = preprocessing(test_data, dict)

    testY = test_data[:,0]
    testX = test_data[:,1:]
    trainY = train_data[:,0]
    trainX = train_data[:,1:]

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
model.add(Dense(220, input_shape=(27,), activation="sigmoid"))
model.add(Dense(11, activation="softmax"))

# train the model using SGDprint("[INFO] training network...")
# opt = SGD(lr=0.2, momentum=0.9)
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss="mean_squared_error", optimizer=opt,metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=25, batch_size=5000)

# evaluate the networkprint("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=5000)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1)))

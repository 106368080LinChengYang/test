import pandas as pd
import numpy as np

# Read dataset into X and Y
df = pd.read_csv('~/HousePrices_Regression/train-v8.csv', delim_whitespace=True, header=None)
dataset = df.values
X_train = dataset[:, 0:18]
y_train = dataset[:, 18]


df2 = pd.read_csv('~/HousePrices_Regression/test-v9.csv', delim_whitespace=True, header=None)
dataset2 = df.values
X_test = dataset[0:6485, 0:18]


from keras.models import Sequential
from keras.layers.core import Dense, Activation



model = Sequential()
model.add(Dense(20, input_dim=18, init='normal', activation='relu'))
model.add(Dense(18, input_dim=20, init='normal', activation='relu'))
# No activation needed in output layer (because regression)
model.add(Dense(1, init='normal'))

# Compile Model
#model.compile(loss='mean_squared_error', optimizer='adam')
model.compile(loss='mae', optimizer='adam')



model.fit(X_train, y_train, epochs=100, batch_size=16,verbose=0)


#score=model.evaluate(X_test, y_test, batch_size=16)


pred=model.predict(X_test)
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


# get dataset
df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

# reduce dataset to desired columns
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# create mask to use 80% of data for training and 20% for test
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# train data
from sklearn import linear_model

# create model object
regr = linear_model.LinearRegression()

# assign x and y (dependant and independent variables)
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# create fit line
regr.fit (train_x, train_y)

# Print the coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

# import r2_score
from sklearn.metrics import r2_score

# create test lists
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# create test predictions
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print(f'R2-score: {r2_score(test_y_hat , test_y):.2f}')
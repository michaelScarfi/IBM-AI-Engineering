import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

# get dataset
df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv")

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# transform dataset so it can be used with linear regression
poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

# create model object
clf = linear_model.LinearRegression()

# create fit line based on transformed dataset
train_y_ = clf.fit(train_x_poly, train_y)

# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

# evaluate model
from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

# transform dataset so it can be used with linear regression
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)

# create model object
clf3 = linear_model.LinearRegression()

# create fit line based on transformed dataset
train_y_3 = clf3.fit(train_x_poly3, train_y)

# The coefficients
print ('Cubic polynomial')
print ('Coefficients: ', clf3.coef_)
print ('Intercept: ',clf3.intercept_)


# evaluate model
from sklearn.metrics import r2_score

test_x_poly3 = poly3.fit_transform(test_x)
test_y_3 = clf3.predict(test_x_poly3)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_3 - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_3 - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_3 , test_y) )
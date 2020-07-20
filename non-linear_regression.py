import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get dataset
df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv")

# define logistical function
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y



# get data
x_data = df["Year"].values
y_data =  df["Value"].values

# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

# using a curve fix to fig sigmoid
from scipy.optimize import curve_fit
# fit curve
popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


# test function

# create mask to use 80% of data for training and 20% for test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# generate Thetas from train data
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# generate curve based off of trained thetas
y_hat = sigmoid(test_x, *popt)

# compare data using thetas to actual data.
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.8f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_hat , test_y) )
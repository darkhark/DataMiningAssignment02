import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pingouin as pg
from sklearn.linear_model import LinearRegression
import seaborn as sbn
pd.set_option('display.max_colwidth', -1)

print("\nReviewing data before analysis\n")

homeData = pd.read_csv("data/HousingPrice.csv")
nan_cols = [i for i in homeData.columns if homeData[i].isnull().any()]
print("Columns with missing data")
print(nan_cols)

print("\nFind columns with > 50% of missing data\n")
print("\tShape before before cleanse" + str(homeData.shape) + "\n")
missingPercentages = homeData.isnull().sum()/len(homeData)
missingPercentages = missingPercentages[missingPercentages > .50]
missingPercentages.sort_values(inplace=True)
print(missingPercentages)

print("\nRemove columns where percent of null is above 50%\n")
for label in missingPercentages.axes:
    homeData = homeData.drop(label, axis=1)
print("Shape after cleanse" + str(homeData.shape))

print("\n----------------Question 1---------------\n")

""" 
Y = B0 + B1 * X
First step in regression analysis is to calculate B1 (scale factor or 
coefficient) and B0 (bias coefficient)
"""
print("Null values in GrLivArea: " + str(homeData['GrLivArea'].isnull().sum()))
x = homeData['GrLivArea'].values
y = homeData['SalePrice'].values

"""
B1 = (Summation of all points with (x - xMean)(y - yMean)) 
        / Summation of all points with (x-xMean)^2
B0 = yMean - B1 * xMean
"""
xMean = np.mean(x)
yMean = np.mean(y)
numOfRows = len(x)
numerator = 0
denominator = 0
for row in range(numOfRows):
    numerator += (x[row] - xMean) * (y[row] - yMean)
    denominator += (x[row] - xMean) ** 2
b1 = numerator/denominator
b0 = yMean - (b1 * xMean)
print(b1, b0)

"""
Plot the regression line for GrLivArea.
"""
# Print values to determine plot sizes
print(np.max(x))
print(np.max(y))
xMax = np.max(x) + 500
xmin = np.min(x) - 500

yMax = np.max(y) + 5000
yMin = np.min(y) - 5000
xLineValues = np.linspace(xmin, xMax, numOfRows)
yLineValues = b0 + b1 * xLineValues
# print(pg.linear_regression(x, y)) Learned after

# Plotting Line
plt.plot(xLineValues, yLineValues, color='#58b970', label='Regression Line')
# Plotting Scatter Points
plt.scatter(x, y, c='#ef5423', label='Scatter Plot')
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.legend()
plt.show()

"""
Compare values from scipy stats with hand calculated values above
"""
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(slope)
print(intercept)

print("\nQuestion 1.1 Answer\n")
print(p_value)

print("\nQuestion 1.2\n")
# pd.options.display.max_columns = 10
# print(pg.corr(x=homeData['GrLivArea'], y=homeData['SalePrice']))
print(slope - 2 * std_err, slope + 2 * std_err)

print("\nQuestion 1.3\n")
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
salesPrediction = model.predict([[2000]])
print(salesPrediction)
salesPrediction = model.predict([[2500]])
print(salesPrediction)

print("\nQuestion 1.4\n")
sbn.residplot(x, y)
plt.xlabel('GrLivArea')
plt.ylabel('Residuals')
plt.show()
print('Residual Plot plotted')

print("\nQuestion 1.5\n")
rSquared = r_value ** 2
print(rSquared)

print("\nQuestion 1.6\n")
print(r_value)

print("\n---------Question 2--------------\n")
pd.options.display.max_columns = 100
dummies = pd.get_dummies(homeData.ExterQual, prefix='ExterQual')
dummies.drop(dummies['ExterQual_TA'])
homeData = pd.concat([homeData, dummies], axis=1)
print(homeData.head())
x1 = homeData['ExterQual_Ex']
x2 = homeData['ExterQual_Fa']
x3 = homeData['ExterQual_Gd']
slope, intercept, r_value, p_value, std_err = stats.linregress(x1, y)
plt.scatter(x1, y, c='#ef5423', label='Scatter Plot')
plt.xlabel("ExterQual_Ex")
plt.ylabel("SalePrice")
plt.legend()
plt.show()

print("\nQuestion 2.1\n")
dataDict = {'ExterQual_Ex': x1, 'ExterQual_Fa': x2, 'ExterQual_Gd': x3}
tempFrame = pd.DataFrame(dataDict)
print(pg.linear_regression(tempFrame, y))
print(slope - 2 * std_err, slope + 2 * std_err)

print("\nQuestion 2.2\n")

model = LinearRegression()
model.fit(tempFrame, y)
print(model.coef_)

print("\nQuestion 2.4\n")
print(model.score(tempFrame, y))

print("\nQuestion 2.6\n")
dataDict = {'ExterQual_Ex': x1, 'ExterQual_Fa': x2}
tempFrame = pd.DataFrame(dataDict)

model = LinearRegression()


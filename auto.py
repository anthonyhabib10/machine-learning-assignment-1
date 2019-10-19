from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd


column = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin"]
# Reads the file then using these attributes it makes a nice table
df = pd.read_csv("auto-mpg.data", names=column, comment="\t", sep=" ", skipinitialspace=True, na_values="?")
# Cleans the data
df = df.fillna(method='ffill')
# plots the correlation so we can determine the best correlation with MPG
print(df.corr(method='spearman'))
# since weight was the closest to -1 we use weight to compare with MPG
x_axis = df[['weight']]
y_axis = df['mpg']

# splits the training data to 80% and test data to 20%
X_train, X_test, y_train, y_test = train_test_split(x_axis, y_axis,
                                                    test_size=0.2,
                                                    random_state=69)
# Plotting the poor model with a degree of 5
degree = PolynomialFeatures(degree=5)
transform = degree.fit_transform(X_train)
regression = LinearRegression()
regression.fit(transform, y_train)
plt.scatter(X_train,y_train)
prediction = regression.predict(transform)
plt.scatter(X_train,prediction, color='blue')


# Plotting better model with a degree of 15
deg = PolynomialFeatures(degree=15)
trans = deg.fit_transform(X_train)
regress = LinearRegression()
regress.fit(trans, y_train)
plt.scatter(X_train,y_train)
predict = regress.predict(trans)
plt.scatter(X_train,predict, color='green')
plt.show()

# calculates MSE
print("The first graph with a degree of 5, the MSE = ", mean_squared_error(prediction, y_train))
print("The second graph with a degree of 25, the MSE = ", mean_squared_error(predict, y_train))
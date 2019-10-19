from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("sonar.all-data", comment="\t", sep=" ", skipinitialspace=True, na_values="?")
df = df.fillna(method='ffill')
print(df)
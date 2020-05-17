# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborm as sns
%matplotlib inline

# Importing the dataset and extact dependent and independent variable
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Visualising the dataset
sns.distplot(dataset['YearsExperience'],kde=False,bins=10)
sns.countplot(y='YearsExperience',data=dataset)
sns.barplot(x='YearsExperience',y='Salary',data=dataset)

# Visualising the data set by drawing correlation map
sns.heatmap(dataset.corr())

# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Finding Residuals
#from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
print('MAE  : ',metrics.mean_absolute_error(y_test, y_pred))
print('MSE  : ',mean_squared_error(y_test, y_pred))
print('RMSE : ',np.sqrt(metrics.mean_absolute_error(y_test, y_pred)))



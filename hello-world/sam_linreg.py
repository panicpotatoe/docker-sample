# Import libraries
import numpy as np #working with arrays
import pandas as pd #dataframe
import matplotlib.pyplot as plt #charts
from sklearn.model_selection import train_test_split #sklearn

def app_run(num):
    # Get the dataset
    data_source = 'https://s3.us-west-2.amazonaws.com/public.gamelab.fun/dataset/salary_data.csv'
    dataset = pd.read_csv(data_source)
    print(f'sample(5)\n{dataset.sample(5)}')

    # Form up X and Y
    X = dataset.iloc[:, :-1].values #get a copy of dataset exclude last column
    y = dataset.iloc[:, 1].values #get array of dataset in column 1st
    print(f'X[:5]\n{X[:5]}')
    print(f'y[:5]\n{y[:5]}')

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
    print(f'len(X_train)\t{len(X_train)}')
    print(f'len(X_test)\t{len(X_test)}')
    print(f'len(y_train)\t{len(y_train)}')
    print(f'len(y_test)\t{len(y_test)}')

    # Fitting Simple Linear Regression to the Training set
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Visualizing the Training set results
    viz_train = plt
    viz_train.scatter(X_train, y_train, color='red')
    viz_train.plot(X_train, regressor.predict(X_train), color='blue')
    viz_train.title('Salary VS Experience (Training set)')
    viz_train.xlabel('Year of Experience')
    viz_train.ylabel('Salary')
    viz_train.show()

    # Visualizing the Test set results
    viz_test = plt
    viz_test.scatter(X_test, y_test, color='red')
    viz_test.plot(X_train, regressor.predict(X_train), color='blue')
    viz_test.title('Salary VS Experience (Test set)')
    viz_test.xlabel('Year of Experience')
    viz_test.ylabel('Salary')
    viz_test.show()

    # Predicting the result of 5 Years Experience
    # expected output: [73545.90445964]
    x_pred01 = num
    y_pred01 = regressor.predict([[x_pred01]])
    print(f'prediction for x={x_pred01} is {y_pred01}')

    # # Predicting the result of [list] of Years Experience
    # x_pred02 = [[5.5], [6.2], [7.1], [7.9]]
    # y_pred02 = regressor.predict(x_pred02)
    # for i in range(0, len(x_pred02)):
    #     print(f'prediction for x={x_pred02[i]} is {y_pred02[i]}')

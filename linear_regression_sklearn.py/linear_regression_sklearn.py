# inspired from Krish Kaik

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset=pd.read_csv('50_Startups.csv')
#separate into dependant and independant features
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

# convert the state column into categorical feature
states=pd.get_dummies(X['State'], drop_first=True).astype(int)

# now we drop the state column and concatanate the dataset with the dummy one
X=X.drop('State',axis=1)
X=pd.concat([X,states],axis=1)

#split the dataset into training and test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)

# create linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# predict the test result
y_pred=regressor.predict(X_test)

#calculate r2 value
# the closer to 1, the better
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)   #the output is 0.935






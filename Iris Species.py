import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

df=pd.read_csv("Iris.csv")
print(df.head())


df_x=df.iloc[:,:-1]
df_y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.3,random_state=4)


dt=DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(dt,x_train,y_train, cv=5)
print("Cross Validation Score of training data is "+str(scores.mean()))


dt.fit(x_train,y_train)
pred=dt.predict(x_test)
print("Predicted Values are "+str(pred))


accuracy=accuracy_score(y_test, pred, normalize=False)
print("Accurately predicted values : " + str(accuracy))
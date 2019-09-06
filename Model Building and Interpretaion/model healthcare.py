import pandas as pd
data=pd.read_excel("D:/Projects/Completed/1.1.Hospital/Model Building and Interpretaion/1555054100_hospitalcosts.xlsx")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.preprocessing import Imputer
obj=Imputer(missing_values="NaN",strategy="most_frequent",axis=0)
obj=obj.fit(x[:,2:4])
x[:,2:4]=obj.transform(x[:,2:4])

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.80,random_state=0)

from sklearn.linear_model import LinearRegression as LR
regressor=LR()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

residuals=y_test-y_pred

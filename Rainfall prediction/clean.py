#Clean the data 
#Importing the important libraries
import pandas as pd
import numpy  as np

import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#read the cleaned data
data = pd.read_csv("austin_final.csv")
X = data.drop(['PrecipitationSumInches'],axis=1)
Y = data['PrecipitationSumInches']
#it will give the values in 2d-vector
Y = Y.values.reshape(-1,1)
#now observing a particular day
day_index=798
days=[i for i in range (Y.size)]

#initialsing the linear regression classiifer
clf = LinearRegression()
# training clf with the output data
clf.fit(X, Y)
#now giving sample input to test our model
inp = np.array([[50],[87],[28.9],[67],[54],[3],[34],[87],[94],[23],[45],[22],[48],[21.90],[87.88],[0],[53]])
#reshaping the input
inp=inp.reshape(1,-1)
#Print the output
print('The precipitation in inches for the input is: ', clf.predict(inp))
#printing the precipitation graph
print("The precipitation trend graph is : ")
plt.scatter(days,Y,color='b')
plt.scatter(days[day_index],Y[day_index],color='r')
plt.title("The precipitaion level")
plt.xlabel("days")
plt.ylabel("Precipitation in inches")
plt.show()
x_vis = X.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent','SeaLevelPressureAvgInches', 'VisibilityAvgMiles','WindAvgMPH'], axis = 1)
print("Precipitaiton vs Selected attribute graph ")
for i in range (x_vis.columns.size):
    plt.subplot(3,2,i+1)
    plt.scatter(days, x_vis[x_vis.columns.values[i][:500]],color = 'g')
  
    plt.scatter(days[day_index], 
                x_vis[x_vis.columns.values[i]][day_index],
                color ='r')
  
    plt.title(x_vis.columns.values[i])
  
plt.show()
'''
#read data in pandas dataframe

#drop uneccessary item from data set

data = data.drop(['Events','Date','SeaLevelPressureHighInches','SeaLevelPressureLowInches'], axis=1)
data=data.replace('T',0.0)
data=data.replace('-',0.0)

#clean version
#saving data
data.to_csv('austin_final.csv')
'''




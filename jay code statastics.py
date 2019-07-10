
#for rice from sklearn
#from osgeo import gdal
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import parser, rrule
from datetime import datetime, time, date
import datetime as tm
import calendar
import seaborn as sns 
import math
from sklearn import datasets
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from datetime import datetime

df['DATE'] = '09/19/18'
DATE=datetime.strptime((df['DATE']),'%m/%d/%y')
datetime_object = datetime.strptime(datetime_str, '%m/%d/%y')

print(type(datetime_object))
print(datetime_object)  # printed in default format

df = pd.read_csv(r"C:\Users\Nitin\name.csv")
X = df['DATE']
Y = df["CLOSE"]
x=np.array(X[:])
y=np.array(Y[:])

x=x.reshape(-1,1)
y=y.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.01, random_state=1)
X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
Y_test=Y_test.reshape(-1,1)
lm=LinearRegression().fit(X_train,Y_train)
predy=lm.predict(X_test)
plt.scatter(Y_test,predy)
plt.scatter(y,Y)
r2_score(Y_test,predy)
print(r2_score)
# t-test
from scipy import stats
t2, p2 = stats.ttest_ind(X,Y)
print("t-test", t2)


# f-test:

import statistics as stats
import scipy.stats as ss
d1=df["HIGH"]
d2=df["LOW"]
def Ftest_pvalue(d1,d2):
    df1 = len(d1) - 1
    df2 = len(d2) - 1
    F = stats.variance(d1) /stats.variance(d2)
    single_tailed_pval = ss.f.cdf(F,df1,df2)
    double_tailed_pval = single_tailed_pval * 2
    return double_tailed_pval
    print("F-test value:",Ftest_pvalue( d1,d2))
ftestvalue=print("F-test value:",Ftest_pvalue( d1,d2))    
plt.plot(y,y1)


#for wheat from sklearn

from osgeo import gdal
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import parser, rrule
from datetime import datetime, time, date
import datetime as tm
import calendar
import seaborn as sns 
import math
from sklearn import datasets
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv(r"C:\Users\student\Tutorial-3.csv")
X = df['Wheat']
Y = df["Temperature"]
x=np.array(X[:])
y=np.array(Y[:])
x=x.reshape(-1,1)
y=y.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1)
X_train=X_train.reshape(-1,1)
Y_train=Y_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
Y_test=Y_test.reshape(-1,1)
lm=LinearRegression().fit(X_train,Y_train)
predy=lm.predict(X_test)
plt.scatter(Y_test,predy)
plt.scatter(y,Y)
r2_score(predy,Y_test)




# t-test
from scipy import stats
t2, p2 = stats.ttest_ind(X,Y)
print("t-test", t2)


# f-test:

import statistics as stats
import scipy.stats as ss
d1=df["Wheat"]
d2=df["Temperature"]
def Ftest_pvalue(d1,d2):
    df1 = len(d1) - 1
    df2 = len(d2) - 1
    F = stats.variance(d1) /stats.variance(d2)
    single_tailed_pval = ss.f.cdf(F,df1,df2)
    double_tailed_pval = single_tailed_pval * 2
    return double_tailed_pval
   
ftestvalue=print("F-test value:",Ftest_pvalue( d1,d2))

#wheat from stats model
df = pd.read_csv(r"C:\Users\student\Tutorial-3.csv")
x1 = df['Wheat']
y1 = df["Temperature"]

x=np.array(x1[:])
y=np.array(y1[:])

y_constant =sm.add_constant(y1)
model=sm.OLS(x1,y1)
lin_reg=model.fit()
lin_reg.summary()
plt.scatter(y,y1)
plt.plot(y,y1)


#rice from stats model

df = pd.read_csv(r"C:\Users\student\Tutorial-3.csv")
x1= df['HIGH']
y1= df["LOW"]
x=np.array(x1[:])
y=np.array(y1[:])

y_constant =sm.add_constant(y1)
model=sm.OLS(x1,y_constant)
lin_reg=model.fit()
lin_reg.summary()
plt.scatter(y,y1)
plt.plot(y,y1)






correlationEP = X_train.corr(Y_train)
a = X_train
b = Y_train
n = np.size(a)
s_x = np.sum(a)
s_y = np.sum(b)
sum1=np.sum(a*b)
sum2 = np.sum(a*a)
sum3 = np.sum(a*a*a)
sum4 = np.sum(a*a*a*a)
sum6 = np.sum(a*a*b)
u = n*sum1 - s_x*s_y
l = n*sum2 - s_x*s_x
A1 = u/l
A0 = (s_y - A1*s_x)/n
z = A0 + A1*X_test
print('Estimated coefficients:\na = {} b={}'.format(A1,A0))
plt.scatter(Y_test,b)
correlationEP = X_train.corr(Y_train)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_st

mehsana.head()
mehsana.info()
mehsana.describe()
mehsana.columns
mehsana.corr()



mehsana1=pd.read_csv(r"C:\Users\student\Mehasana_EC_pH_f.csv")

mehsana=pd.read_csv(r"C:\Users\student\Multiple_LR.csv")


#for rice from sklearn

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\student\Tutorial-3.csv")
X = df["Rainfall"]
Y = df['Rice']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1, random_state=1)
correlationEP = X_train.corr(Y_train)
a = X_train
b = Y_train
n = np.size(a)
s_x = np.sum(a)
s_y = np.sum(b)
sum1=np.sum(a*b)
sum2 = np.sum(a*a)
sum3 = np.sum(a*a*a)
sum4 = np.sum(a*a*a*a)
sum6 = np.sum(a*a*b)
u = n*sum1 - s_x*s_y
l = n*sum2 - s_x*s_x
A1 = u/l
A0 = (s_y - A1*s_x)/n
z = A0 + A1*X_test
print('Estimated coefficients:\na = {} b={}'.format(A1,A0))
r2_score(z,Y_test)




plt.scatter(Y_test,b)
plt.scatter(X_test,b)
mat = np.array([[n,s_x], [s_x,sum2]])
mat3 = np.array([[s_y], [sum1]])
inv = np.linalg.inv(mat)
malt = np.matmul(inv, mat3)
y_pred = malt[0] + malt[1]*X_test
r2_score(Y,X_test)
mat4 = np.array([[n,s_x,sum2],[s_x, sum2, sum3], [sum2, sum3, sum4]])
inv2= np.linalg.inv(mat4)
mat5 = np.array([[s_y], [sum5], [sum6]])
malt1 = np.matmul(inv2,mat5)
y_pred = malt1[0] + malt1[1]*X_test + malt1[2]*X_test*X_test
plt.scatter(X_test,y_pred)
plt.scatter(y_pred,Y_test)
r2_score(Y,y_pred)

sklearn.metrics.r2_score(Y, y_pred, sample_weight=None, multioutput=’uniform_average’)





  
#for reading dATE COLUMNS
parse_dates=True 
parse_dates=['column name']
DF= pd.read_csv(r"C:\Users\Nitin\name.csv")
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
DF = pd.read_csv(df, parse_dates=['DATE'], date_parser=dateparse)


D = {'date': '2013-6-4'}
df = pd.DataFrame(D, index=[0])
df
df.dtypes
df['DATE'] = pd.to_datetime(df.date, format='%Y-%m-%d')

df1=DF.dtypes['DATE']
X.append(df1)


#diference between predict and y-test value
diff=predy-Y_test
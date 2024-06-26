# 123OFAI_California_Linear_Regressor
Projects during the 123OfAI AlphaML Course - Implement a Linear Regressor on the California Housing Dataset

```python
#Import Necessary Libraries:
import pandas as pd
import numpy as np
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
```


```python
#1. Load Data
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing(as_frame=True)
print(california_housing.DESCR)
```

    .. _california_housing_dataset:
    
    California Housing dataset
    --------------------------
    
    **Data Set Characteristics:**
    
    :Number of Instances: 20640
    
    :Number of Attributes: 8 numeric, predictive attributes and the target
    
    :Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude
    
    :Missing Attribute Values: None
    
    This dataset was obtained from the StatLib repository.
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    
    The target variable is the median house value for California districts,
    expressed in hundreds of thousands of dollars ($100,000).
    
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    
    A household is a group of people residing within a home. Since the average
    number of rooms and bedrooms in this dataset are provided per household, these
    columns may take surprisingly large values for block groups with few households
    and many empty houses, such as vacation resorts.
    
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    
    .. topic:: References
    
        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297
    
    


```python
california_housing.frame.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseVal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>




```python
california_housing.data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
    </tr>
  </tbody>
</table>
</div>




```python
import math
print(math.log(452600))
```

    13.022764012181574
    


```python
california_housing.data.columns
```




    Index(['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
           'Latitude', 'Longitude'],
          dtype='object')




```python
print(california_housing.data.shape)
print(california_housing.target.shape)
```

    (20640, 8)
    (20640,)
    


```python
california_housing.feature_names
```




    ['MedInc',
     'HouseAge',
     'AveRooms',
     'AveBedrms',
     'Population',
     'AveOccup',
     'Latitude',
     'Longitude']




```python
pd.set_option('display.precision', 4)
pd.set_option('display.max_columns', 9)
pd.set_option('display.width', None)
california_df = pd.DataFrame(california_housing.data,
                             columns=california_housing.feature_names)
california_df['MedHouseValue'] = pd.Series(california_housing.target)
california_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.9841</td>
      <td>1.0238</td>
      <td>322.0</td>
      <td>2.5556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.2381</td>
      <td>0.9719</td>
      <td>2401.0</td>
      <td>2.1098</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.2881</td>
      <td>1.0734</td>
      <td>496.0</td>
      <td>2.8023</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.8174</td>
      <td>1.0731</td>
      <td>558.0</td>
      <td>2.5479</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.2819</td>
      <td>1.0811</td>
      <td>565.0</td>
      <td>2.1815</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>




```python
california_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.0000</td>
      <td>20640.0000</td>
      <td>20640.0000</td>
      <td>20640.0000</td>
      <td>20640.0000</td>
      <td>20640.0000</td>
      <td>20640.0000</td>
      <td>20640.0000</td>
      <td>20640.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.8707</td>
      <td>28.6395</td>
      <td>5.4290</td>
      <td>1.0967</td>
      <td>1425.4767</td>
      <td>3.0707</td>
      <td>35.6319</td>
      <td>-119.5697</td>
      <td>2.0686</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.8998</td>
      <td>12.5856</td>
      <td>2.4742</td>
      <td>0.4739</td>
      <td>1132.4621</td>
      <td>10.3860</td>
      <td>2.1360</td>
      <td>2.0035</td>
      <td>1.1540</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.4999</td>
      <td>1.0000</td>
      <td>0.8462</td>
      <td>0.3333</td>
      <td>3.0000</td>
      <td>0.6923</td>
      <td>32.5400</td>
      <td>-124.3500</td>
      <td>0.1500</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.5634</td>
      <td>18.0000</td>
      <td>4.4407</td>
      <td>1.0061</td>
      <td>787.0000</td>
      <td>2.4297</td>
      <td>33.9300</td>
      <td>-121.8000</td>
      <td>1.1960</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.5348</td>
      <td>29.0000</td>
      <td>5.2291</td>
      <td>1.0488</td>
      <td>1166.0000</td>
      <td>2.8181</td>
      <td>34.2600</td>
      <td>-118.4900</td>
      <td>1.7970</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.7432</td>
      <td>37.0000</td>
      <td>6.0524</td>
      <td>1.0995</td>
      <td>1725.0000</td>
      <td>3.2823</td>
      <td>37.7100</td>
      <td>-118.0100</td>
      <td>2.6472</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.0001</td>
      <td>52.0000</td>
      <td>141.9091</td>
      <td>34.0667</td>
      <td>35682.0000</td>
      <td>1243.3333</td>
      <td>41.9500</td>
      <td>-114.3100</td>
      <td>5.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#2. Handle missing values :

california_df.isnull().sum()
```




    MedInc           0
    HouseAge         0
    AveRooms         0
    AveBedrms        0
    Population       0
    AveOccup         0
    Latitude         0
    Longitude        0
    MedHouseValue    0
    dtype: int64




```python
#3. Standardize data :
# Get column names first
names = california_df.columns
# Create the Scaler object
scaler = StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(california_df)
scaled_df = pd.DataFrame(scaled_df, columns=names)
scaled_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.3448</td>
      <td>0.9821</td>
      <td>0.6286</td>
      <td>-0.1538</td>
      <td>-0.9744</td>
      <td>-0.0496</td>
      <td>1.0525</td>
      <td>-1.3278</td>
      <td>2.1296</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.3322</td>
      <td>-0.6070</td>
      <td>0.3270</td>
      <td>-0.2633</td>
      <td>0.8614</td>
      <td>-0.0925</td>
      <td>1.0432</td>
      <td>-1.3228</td>
      <td>1.3142</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.7827</td>
      <td>1.8562</td>
      <td>1.1556</td>
      <td>-0.0490</td>
      <td>-0.8208</td>
      <td>-0.0258</td>
      <td>1.0385</td>
      <td>-1.3328</td>
      <td>1.2587</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.9330</td>
      <td>1.8562</td>
      <td>0.1570</td>
      <td>-0.0498</td>
      <td>-0.7660</td>
      <td>-0.0503</td>
      <td>1.0385</td>
      <td>-1.3378</td>
      <td>1.1651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.0129</td>
      <td>1.8562</td>
      <td>0.3447</td>
      <td>-0.0329</td>
      <td>-0.7598</td>
      <td>-0.0856</td>
      <td>1.0385</td>
      <td>-1.3378</td>
      <td>1.1729</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 4. visualize relationship between features and target :
scaled_df.columns
```




    Index(['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
           'Latitude', 'Longitude', 'MedHouseValue'],
          dtype='object')




```python
#plot graphs
fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='Longitude',y='MedHouseValue',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='Latitude',y='MedHouseValue',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='HouseAge',y='MedHouseValue',ax=axs[2],figsize=(16,8))

#plot graphs
fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='AveRooms',y='MedHouseValue',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='AveBedrms',y='MedHouseValue',ax=axs[1],figsize=(16,8))
scaled_df.plot(kind='scatter',x='Population',y='MedHouseValue',ax=axs[2],figsize=(16,8))

#plot graphs
fig,axs=plt.subplots(1,3,sharey=True)
scaled_df.plot(kind='scatter',x='AveOccup',y='MedHouseValue',ax=axs[0],figsize=(16,8))
scaled_df.plot(kind='scatter',x='MedInc',y='MedHouseValue',ax=axs[1],figsize=(16,8))
```




    <Axes: xlabel='MedInc', ylabel='MedHouseValue'>




    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_13_1.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_13_2.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_13_3.png)
    



```python
for column in scaled_df:
    plt.figure()
    sns.boxplot(x=scaled_df[column])
```


    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_0.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_1.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_2.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_3.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_4.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_5.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_6.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_7.png)
    



    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_14_8.png)
    



```python
#5.Extract X and Y data :

X_Features=['Longitude', 'Latitude', 'HouseAge', 'AveRooms',
       'AveBedrms', 'Population', 'AveOccup', 'MedInc']
X=scaled_df[X_Features]
Y=scaled_df['MedHouseValue']

print(type(X))
print(type(Y))
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.series.Series'>
    


```python
print(california_df.shape)
print(X.shape)
print(Y.shape)
```

    (20640, 9)
    (20640, 8)
    (20640,)
    


```python
#6. Split the dataset :
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

print (x_train.shape, y_train.shape)
print (x_test.shape, y_test.shape)
```

    (16512, 8) (16512,)
    (4128, 8) (4128,)
    


```python
#7. Apply Various Algorithms:
#1.Linear Regression
#2.Decision Tree Regression
#3.Random Forest Regression (Ensemble Learning)
#4.Lasso
#5.Ridge
#6.Elastic Net
```


```python
# 1.Perform Linear Regression :

##### Perform Linear Regression on training data.
#####Predict output for test dataset using the fitted model.
#####Print root mean squared error (RMSE) from Linear Regression.

linreg=LinearRegression()
linreg.fit(x_train,y_train)
```


```python
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)
```



```python
y_predict = linreg.predict(x_test)
```


```python
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))
```

    0.6303860650696056
    0.5965968374812354
    


```python
#2. Perform Decision Tree Regression :

####Perform Decision Tree Regression on training data.
####Predict output for test dataset using the fitted model.
####Print root mean squared error from Decision Tree Regression.

dtreg=DecisionTreeRegressor()
dtreg.fit(x_train,y_train)
```


```python
DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,
                      random_state=None, splitter='best')
```




```python
y_predict = dtreg.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))
```

    0.6143097660433482
    0.6169098997828322
    


```python
#3. Perform Random Forest Regression :
####Perform Random Forest Regression on training data.
####Predict output for test dataset using the fitted model.
####Print RMSE (root mean squared error) from Random Forest Regression.
from sklearn.ensemble import RandomForestRegressor
rfreg=RandomForestRegressor()
rfreg.fit(x_train,y_train)
```


```python
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=10,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
```


```python
y_predict = rfreg.predict(x_test)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))
```

    0.4392038830340111
    0.8041793751284102
    


```python
#4. Perform Lasso Regression (determine which variables should be retained in the model):

####Perform Lasso Regression on training data.
####Predict output for test dataset using the fitted model.
####Print RMSE (root mean squared error) from Lasso Regression.


lassoreg=Lasso(alpha=0.001)
lassoreg.fit(x_train,y_train)
print(sqrt(mean_squared_error(y_test,lassoreg.predict(x_test))))
print('R2 Value/Coefficient of determination:{}'.format(lassoreg.score(x_test,y_test)))
```

    0.630326757504838
    R2 Value/Coefficient of determination:0.5966727393295559
    


```python
#5. Perform Ridge Regression (addresses multicollinearity issues) :

#Perform Ridge Regression on training data.
#Predict output for test dataset using the fitted model.
#Print RMSE (root mean squared error) from Ridge Regression.

ridgereg=Ridge(alpha=0.001)
ridgereg.fit(x_train,y_train)
print(sqrt(mean_squared_error(y_test,ridgereg.predict(x_test))))
print('R2 Value/Coefficient of determination:{}'.format(ridgereg.score(x_test,y_test)))
```

    0.6303860579667828
    R2 Value/Coefficient of determination:0.5965968465718587
    


```python
#6. Perform ElasticNet Regression :

#Perform ElasticNet Regression on training data.
#Predict output for test dataset using the fitted model.
#Print RMSE (root mean squared error) from ElasticNet Regression.

#from sklearn.linear_model import ElasticNet
elasticreg=ElasticNet(alpha=0.001)
elasticreg.fit(x_train,y_train)
print(sqrt(mean_squared_error(y_test,elasticreg.predict(x_test))))
print('R2 Value/Coefficient of determination:{}'.format(elasticreg.score(x_test,y_test)))
```

    0.6303027367217616
    R2 Value/Coefficient of determination:0.5967034791067273
    


```python

lm=smf.ols(formula='MedHouseValue ~ Longitude+Latitude+HouseAge+AveRooms+AveBedrms+Population+AveOccup+MedInc',data=scaled_df).fit()
```


```python
lm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>MedHouseValue</td>  <th>  R-squared:         </th> <td>   0.606</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.606</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3970.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 02 Apr 2024</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>08:41:15</td>     <th>  Log-Likelihood:    </th> <td> -19669.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20640</td>      <th>  AIC:               </th> <td>3.936e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20631</td>      <th>  BIC:               </th> <td>3.943e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     8</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>  <td> 2.663e-16</td> <td>    0.004</td> <td>  6.1e-14</td> <td> 1.000</td> <td>   -0.009</td> <td>    0.009</td>
</tr>
<tr>
  <th>Longitude</th>  <td>   -0.7544</td> <td>    0.013</td> <td>  -57.682</td> <td> 0.000</td> <td>   -0.780</td> <td>   -0.729</td>
</tr>
<tr>
  <th>Latitude</th>   <td>   -0.7798</td> <td>    0.013</td> <td>  -58.541</td> <td> 0.000</td> <td>   -0.806</td> <td>   -0.754</td>
</tr>
<tr>
  <th>HouseAge</th>   <td>    0.1029</td> <td>    0.005</td> <td>   21.143</td> <td> 0.000</td> <td>    0.093</td> <td>    0.112</td>
</tr>
<tr>
  <th>AveRooms</th>   <td>   -0.2301</td> <td>    0.013</td> <td>  -18.235</td> <td> 0.000</td> <td>   -0.255</td> <td>   -0.205</td>
</tr>
<tr>
  <th>AveBedrms</th>  <td>    0.2649</td> <td>    0.012</td> <td>   22.928</td> <td> 0.000</td> <td>    0.242</td> <td>    0.288</td>
</tr>
<tr>
  <th>Population</th> <td>   -0.0039</td> <td>    0.005</td> <td>   -0.837</td> <td> 0.402</td> <td>   -0.013</td> <td>    0.005</td>
</tr>
<tr>
  <th>AveOccup</th>   <td>   -0.0341</td> <td>    0.004</td> <td>   -7.769</td> <td> 0.000</td> <td>   -0.043</td> <td>   -0.025</td>
</tr>
<tr>
  <th>MedInc</th>     <td>    0.7190</td> <td>    0.007</td> <td>  104.054</td> <td> 0.000</td> <td>    0.705</td> <td>    0.732</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>4393.650</td> <th>  Durbin-Watson:     </th> <td>   0.885</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>14087.596</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.082</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.420</td>  <th>  Cond. No.          </th> <td>    6.67</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
# Perform Linear Regression with one independent variable :

#Extract just the median_income column from the independent variables (from X_train and X_test).
#Perform Linear Regression to predict housing values based on median_income.
#Predict output for test dataset using the fitted model.
#Plot the fitted model for training data as well as for test data to check if the fitted model satisfies the test data.
```


```python
x_train_Income=x_train[['MedInc']]
x_test_Income=x_test[['MedInc']]
```


```python
print(x_train_Income.shape)
print(y_train.shape)
```

    (16512, 1)
    (16512,)
    


```python
# visualize relationship between features


linreg=LinearRegression()
linreg.fit(x_train_Income,y_train)
y_predict = linreg.predict(x_test_Income)
```


```python
#print intercept and coefficient of the linear equation
print(linreg.intercept_, linreg.coef_)
print(sqrt(mean_squared_error(y_test,y_predict)))
print((r2_score(y_test,y_predict)))
```

    0.0056230198668934615 [0.69238221]
    0.7212595914243148
    0.4719083593446771
    


```python
#Insight:
#Looking at the above values we can say that coefficient: a unit increase in median_income increases the median_house_value by 0.692 unit
```


```python
#plot least square line
scaled_df.plot(kind='scatter',x='MedInc',y='MedHouseValue')
plt.plot(x_test_Income,y_predict,c='red',linewidth=2)
```




    [<matplotlib.lines.Line2D at 0x268fae45d90>]




    
![png](California_Dataset_Linear_Regression_files/California_Dataset_Linear_Regression_40_1.png)
    



```python
# Hypothesis testing and P values:
"""using the null hypothesis lets assume there is no relationship between median_income and median_house_value
Lets test this hypothesis. We shall reject the Null Hypothesis if 95% confidence inderval does not include 0"""
```


```python
lm=smf.ols(formula='MedHouseValue ~ MedInc',data=scaled_df).fit()
```


```python
lm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>MedHouseValue</td>  <th>  R-squared:         </th> <td>   0.473</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.473</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>1.856e+04</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 02 Apr 2024</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>09:15:01</td>     <th>  Log-Likelihood:    </th> <td> -22668.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 20640</td>      <th>  AIC:               </th> <td>4.534e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 20638</td>      <th>  BIC:               </th> <td>4.536e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td> 4.753e-16</td> <td>    0.005</td> <td> 9.41e-14</td> <td> 1.000</td> <td>   -0.010</td> <td>    0.010</td>
</tr>
<tr>
  <th>MedInc</th>    <td>    0.6881</td> <td>    0.005</td> <td>  136.223</td> <td> 0.000</td> <td>    0.678</td> <td>    0.698</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>4245.795</td> <th>  Durbin-Watson:     </th> <td>   0.655</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>9273.446</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.191</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.260</td>  <th>  Cond. No.          </th> <td>    1.00</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
#INSIGHT:
"""The P value is 0.000 indicates strong evidence against the null hypothesis, so you reject the null hypothesis.
 so, there is a strong relationship between median_house_value and median_income"""
```


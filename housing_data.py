#==================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#==================================================================================
#===================================================================================
housing_data = pd.read_csv("housing.csv")
#===================================================================================
#===============================================================================================================================
X = housing_data[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income','ocean_proximity']]
y = housing_data[['median_house_value']]
#===============================================================================================================================

#======================================================================================================
from sklearn.preprocessing import Imputer
imp_housing = Imputer(missing_values="NaN",strategy="mean",axis=0)
X[['total_bedrooms']] = imp_housing.fit_transform(X[['total_bedrooms']])
#======================================================================================================
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lbl_encoder = LabelEncoder()
X[['ocean_proximity']] = lbl_encoder.fit_transform(X[['ocean_proximity']])
onehotencoder_X = OneHotEncoder(categorical_features=[8])
X = onehotencoder_X.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:,[7,12]] = sc_X.fit_transform(X_train[:,[7,12]])
X_test[:,[7,12]] = sc_X.transform(X_test[:,[7,12]])
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)
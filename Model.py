#%%
import pandas as pd
import numpy as np
from sklearn import pipeline,preprocessing,metrics,model_selection,ensemble
from sklearn_pandas import DataFrameMapper
# %%
data=pd.read_csv('Data/mpg_data_example.csv')
# %%
data.head()

# %%
data.isnull().sum()

# %%

mapper = DataFrameMapper([
                        (['cylinders','displacement','weight','acceleration','model year'], preprocessing.StandardScaler()),
                        (['horsepower'],preprocessing.Imputer()),
                        (['origin'], preprocessing.OneHotEncoder())
                        ])

# %%
pipeline_obj = pipeline.Pipeline([
    ('mapper',mapper),
    ("model", ensemble.RandomForestRegressor())
])

# %%
data.columns

# %%
X=['cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'model year', 'origin']
Y=['mpg']

# %%
pipeline_obj.fit(data[X],data[Y])

# %%
pipeline_obj.predict(data[X])

# %%
from sklearn.externals import joblib

# %%
joblib.dump(pipeline_obj,'RFModelforMPG.pkl')

# %%
modelReload=joblib.load('RFModelforMPG.pkl')

# %%
modelReload.predict(data[X])

# %%
temp={}
temp['cylinders']=1
temp['displacement']=2
temp['horsepower']=3
temp['weight']=4
temp['acceleration']=5
temp['model year']=6
temp['origin']=1

# %%
testDtaa=pd.DataFrame({'x':temp}).transpose()

# %%
testDtaa

# %%
modelReload.predict(testDtaa)[0]
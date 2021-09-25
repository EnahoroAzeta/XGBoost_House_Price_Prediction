#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from google.colab import files
uploaded = files.upload()

import io
train = pd.read_csv(io.BytesIO(uploaded['train.csv']))
test = pd.read_csv(io.BytesIO(uploaded['test.csv']))



#Define features and target
X = train.copy()
y = X.pop('SalePrice')
X_test = test.copy()

# Data split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# All categorical columns, training and validation
cat_cols = [col for col in X.columns if X[col].dtype == "object"]

# All numeric columns
num_cols = list(set(X.columns)-set(cat_cols))
num_cols.remove('Id')

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

model = XGBRegressor(n_estimators=500, early_stopping_rounds=5,eval_set=[(X_valid, y_valid)], verbose=False)


# In[ ]:


pip install xgboost --no-binary xgboost -v


# In[ ]:


files.upload()


# In[3]:




# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)


# In[4]:


result = my_pipeline.predict(X_test)

 

result_dataframe = pd.DataFrame(result, columns=['predictions']) 

test_id = test['Id']

result_dataframe


prediction = pd.concat([test_id, result_dataframe], axis=1)
prediction

prediction_int = prediction.astype({"predictions": int})

prediction_int.rename(columns={'predictions': 'SalePrice'}, inplace=True)


from google.colab import drive
drive.mount('drive')

prediction_int.to_csv('/content/drive/My Drive/predictions4.csv', encoding='utf-8', index=False)


# In[ ]:


scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)


# In[ ]:


print("Average MAE score (across experiments):")
print(scores.mean())


# In[ ]:





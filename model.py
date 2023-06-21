#importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

#importing dataset
le_df = pd.read_csv('Life Expectancy Data.csv')

#dropping unwanted columns
le_df.drop(['Year', 'Status'], axis=1, inplace=True)

#renaming columns
le_df.rename(columns={'Life expectancy':'Life Expectancy', 'infant deaths':'Infant Deaths',
                      'percentage expenditure':'Percentage Expenditure',
                      'under-five deaths':'Under-Five Deaths',
                     'thinness  1-19 years':'Thinness 10-19 years',
                      'thinness 5-9 years':'Thinness 5-9 years'}, inplace=True)

#le_df.isnull().head()
#total = le_df.isnull().sum()
#total
numeric_data = le_df.select_dtypes(include=np.number)
numeric_col = numeric_data.columns
for i in numeric_col:
    mean = le_df[i].mean()
    le_df[i].fillna(mean,inplace = True)
le_df = le_df.groupby('Country').mean()
#le_df

#splitting into dependant & independant variables
life = le_df['Life Expectancy']
features = le_df.drop(['Life Expectancy'], axis=1)

#splitting into train & test
X_train, X_test, y_train, y_test = train_test_split(features, life, test_size = 0.2, random_state = 0)

#training
ran_forest_reg = RandomForestRegressor()
ran_forest_reg.fit(X_train, y_train)

#Saving model to disk
pickle.dump(ran_forest_reg, open('model.pkl','wb'))

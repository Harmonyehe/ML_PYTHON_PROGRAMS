import pandas as pd
data=pd.read_csv('dataset.csv')
data.head()
data.drop('Birthplace',axis=1)
data.groupby('Sex').agg({'Political_party':'count'})
data.iloc[:5,8]
desc=data['Name'].describe()
desc
data.drop(2,axis=0)
data.rename(columns={'Account_ID':'ID'})
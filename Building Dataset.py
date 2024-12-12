import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
 
data = pd.read_csv('train.csv') 
 
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) 

data['Age'].fillna(data['Age'].mean()) 
data['Embarked'].fillna(data['Embarked'].mode()[0]) 

X = data.drop('Survived', axis=1) 
y = data['Survived'] 

encoder = LabelEncoder() 
categorical_cols = ['Sex', 'Embarked'] 
for col in categorical_cols: 
    X[col] = encoder.fit_transform(X[col]) 

scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
random_state=42) 

print("Training set shape:", X_train.shape) 
print("Testing set shape:", X_test.shape)
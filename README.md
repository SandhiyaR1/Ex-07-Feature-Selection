# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file
# CODE
```
#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# data loading
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()

#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(data.isnull(),cbar=False)

#Data Cleaning and Data Drop Process
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())

# Change to categoric column to numeric
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1

# instead of nan values
data['Embarked']=data['Embarked'].fillna('S')

# Change to categoric column to numeric
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2

#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)

data.head(11)

#heatmap for train dataset
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

# Now, data is clean and read to a analyze
sns.heatmap(data.isnull(),cbar=False)

# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

#Age with survived
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()

#Count the pessenger class
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```

# OUPUT
![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/c0a3e653-d47a-4755-a7e9-e8380d28a240)

![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/4b69f223-17e5-4f78-966d-d25b78b350f4)
![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/f94bd23a-5158-497d-a078-8b38e485742f)

![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/0d3315ac-6ef6-43fc-b7c8-1bef23b1973d)

![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/7b7c5622-5a81-4967-9cb7-fae2eb348afa)
![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/5b46e116-7009-41ef-b4d4-6ce9edb6d9cc)
![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/aed417c6-6270-4c52-80bd-3e6b04d43b39)
![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/4fa6427a-b4e3-4547-848a-870948378a85)

![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/70d84ef9-2564-4ee3-a15a-dec20c0372d8)

![image](https://github.com/SandhiyaR1/Ex-07-Feature-Selection/assets/113497571/832726cb-5488-4b37-b673-bf9d1587940a)
# RESULT:
Thus, Sucessfully performed the various feature selection techniques on a given dataset.




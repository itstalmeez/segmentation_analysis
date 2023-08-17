# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("Mall_Customers.csv")
print(df)
df.info()
print("\n\n\n NO missing values")
#Gender is an object. lets convert it into categorical feature using pd.dummies(one hot encode)

#The advantages of using one hot encoding include:
#It allows the use of categorical variables in models that require numerical input.
#It can improve model performance by providing more information to the model about the categorical variable.

df['Gender'] = pd.get_dummies(df['Gender'],drop_first=True)
#one hot encodig successfull as we can see in the dtype
df.info()
import matplotlib.pyplot as plt
import seaborn as sns

# 1.  There is no strong coorelation between any columns ( using heatmap and pairplot).

# 2. As age of both gender increases their spending habit decreases.

# 3. female      mean Age 38.098214   mean Annual Income (k$)59.250000     mean Spending Score 51.526786
# 4. male        mean Age 39.806818   mean Annual Income (k$)62.227273     mean Spending Score 48.511364

# 5. max income of female is 126k and min is 16k
# 6. max income of male is 137k and min is 15k

# 7. highest spending score is of a female (99 out of 100)
# lets see the plots between different columns of the dataset
sns.pairplot(df)
sns.heatmap(df.corr(),annot=True)
#outliers  0 = Female , 1= Male
sns.boxplot(x='Gender',y='Age',data=df);
sns.scatterplot(data=df,x='Age',y='Spending Score (1-100)',hue='Gender')
plt.title("Blue is female and orange is Male")
plt.show()
Gen =df.groupby('Gender')
print("\t\t\t0 is female and 1 is male")
Gen.mean()
print(Gen.max())
print('\n\n')
print(Gen.min())
X= df.iloc[:, [3,4]].values
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i)
    km.fit_predict(X)
    wcss.append(km.inertia_)

km = KMeans(n_clusters=5)
y_means = km.fit_predict(X)
plt.scatter(X[y_means == 0,0],X[y_means == 0,1],color='blue')
plt.scatter(X[y_means == 1,0],X[y_means == 1,1],color='red')
plt.scatter(X[y_means == 2,0],X[y_means == 2,1],color='green')
plt.scatter(X[y_means == 3,0],X[y_means == 3,1],color='yellow')
plt.scatter(X[y_means == 4,0],X[y_means == 4,1],color='black')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
model=KMeans(n_clusters=5)
model.fit(df)
pre=model.predict(df)
df["Target"]=y_means
df=df
df.head()
X=df.iloc[:,1:5]
y=df.iloc[:,-1]
X.head()
y.head()
#splitting the dataset
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#Standardize the varriables
from sklearn.preprocessing import StandardScaler
Sc = StandardScaler()
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
error_rate=[]

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))
plt.figure(figsize=(10,5))
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o',markersize=12)
plt.title("Error rate vs k value")
plt.xlabel("k")
plt.ylabel("Error_rate")
plt.show()
knn =KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
pred_5=knn.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, pred_5)
accuracy
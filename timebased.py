
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

data = pd.read_csv('C:/Users/PRIYANSHU SHARMA/Desktop/PRIYANSHU/6 STUDY/6 SEMSTER/LARGE SCALE DATA PROCESSING/PROJECT/netfli.csv', na_values=['NA'])


# In[3]:

data.head()


# In[4]:

data.describe()


# In[5]:

data['user rating score'].fillna((data['user rating score'].mean()), inplace=True)


# In[6]:

data.info()


# In[7]:

data.head()


# In[8]:

print(data.columns)


# In[9]:

rating_mapping = {'TV-MA':9, 'R':10, 'NR':0, 'UR':11, 'TV-14':4, 'PG-13':5, 'TV-PG':12, 'PG':6, 'TV-Y7':1, 'TV-Y7-FV':7, 'TV-Y':2, 'TV-G':8, 'G':3}
data['rating'] = data['rating'].map(rating_mapping)


# In[10]:

import numpy as np
import scipy as sp
data.dropna(subset=['ratingLevel'])


# In[11]:

data.info()


# In[12]:

for i in range(1,1001):
    kwargs = { 'score_size': lambda x: (x['user rating score'])/x['user rating size']}
    data= data.assign(**kwargs)
data.head()


# In[13]:

from textblob import TextBlob


# In[14]:

def textsubjectivity(Text):
    a = TextBlob(Text)
    b = a.sentiment.polarity
    sub = a.sentiment.subjectivity
    return sub


# In[15]:

def textpolarity(Text):
    a = TextBlob(Text)
    pol = a.sentiment.polarity
    return pol


# In[16]:

subar=[]
polar=[]
for i in range(0,1000):
    Text = data['ratingLevel'][i]
    s = textsubjectivity(Text)
    subar.append(s)
    p = textpolarity(Text)
    polar.append(p)


# In[23]:

subar = np.asarray(subar)
polar = np.asarray(polar)
for i in range(0,1000):
#     print(subar[i])
     data['tsub'][i] = subar[i]
#     print(data['tsub'][i])
data.head()


# In[27]:

for i in range(0,1000):
#     print(polar[i])
     data['tpol'][i] = polar[i]
#    print(data['tpol'][i])
data.head()


# In[199]:

del data['f_class']
data.head()


# In[200]:

data.to_csv('C:/Users/PRIYANSHU SHARMA/Desktop/PRIYANSHU/6 STUDY/6 SEMSTER/LARGE SCALE DATA PROCESSING/PROJECT/netfli_main.csv')


# In[308]:

import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
cols = ['rating', 'ratingDescription', 'release year', 'score_size', 'tsub' , 'tpol']
cm = np.corrcoef(data[cols].values.T)
sb.set(font_scale=1)
hm=sb.heatmap(cm,
               cbar=True,
               annot=True,
               square=True,
               fmt='.2f',
               annot_kws={'size':10},
               yticklabels=cols,
               xticklabels=cols)
plt.show()


# In[309]:

data.loc[data['rating']<4,'class']= 'For All'
data.loc[((data['rating']>=4) & (data['rating']<9)),'class']= 'Parential Guidance'
data.loc[data['rating']>=9,'class']= 'For Above 17'


# In[310]:

data.head()


# In[311]:

data.tail()


# In[312]:

from sklearn.cross_validation import train_test_split
X = data.iloc[:, [3,4,5,6,8,9]].values
X_train,X_test,y_train,y_test = train_test_split(X,data['class'], test_size=0.2, random_state=0)


# In[313]:

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.2, C=10.0)
svm.fit(X_train, y_train)


# In[314]:

svm.score(X_test, y_test, sample_weight=None)


# In[315]:

comm = "Parental guidance suggested. May not be suitable for all children."
a = TextBlob(comm)
a_sub = a.sentiment.subjectivity
a_pol = a.sentiment.polarity
print(a_sub)
print(a_pol)


# In[316]:

x = [[74,2015,96,83,a_sub,a_pol]] 
svm.predict(x)


# In[210]:

import matplotlib.pyplot as plt
import seaborn as sb
cols = ['rating', 'ratingDescription', 'release year', 'score_size' , 'tsub' , 'tpol']


# In[211]:

sb.pairplot(data.dropna(),hue='class')


# In[212]:

plt.show()


# In[213]:

from sklearn.cross_validation import train_test_split
X = data.iloc[:, [3,7]].values
X_train,X_test,y_train,y_test = train_test_split(X,data['class'], test_size=0.3, random_state=0)


# In[214]:

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.2, C=10.0)
svm.fit(X_train, y_train)


# In[215]:

svm.score(X_test, y_test, sample_weight=None)


# In[216]:

print(data.columns)


# In[217]:

#DECISION TREE


# In[218]:

simple_mapping = {'For All':0, 'Parential Guidance':1, 'For Above 17':2}
data['f_class'] = data['class'].map(simple_mapping)
data.head()


# In[219]:

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[220]:

columns = ['ratingDescription', 'release year','score_size', 'tsub', 'tpol']
X_train,X_test,Y_train,Y_test = train_test_split(data[columns],data['class'],test_size=0.2,random_state=14)
tree = DecisionTreeClassifier(max_depth=7,random_state=0)
tree.fit(X_train,Y_train)


# In[221]:

print("Accuracy on the training set: %.3f" % tree.score(X_train,Y_train))
print("Accuracy on the testing set: %.3f" % tree.score(X_test,Y_test))


# In[222]:

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=['1','2','3'], impurity=False, filled=True,
                feature_names=data[columns].columns)


# In[223]:

import graphviz


# In[224]:

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# In[326]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

feat_labels = data.columns[[3,4,7,8,9]]
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(X_train, Y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(0,5):
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[f],importances[indices[f]]))


# In[330]:

import matplotlib.pyplot as plt
plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),importances[indices],color='darkred',align='center')
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.figure(figsize=(8,6))
plt.show()


# In[239]:

columns = ['ratingDescription', 'score_size']
X_train,X_test,Y_train,Y_test = train_test_split(data[columns],data['f_class'],test_size=0.2,random_state=14)
data.head()


# In[241]:

from sklearn.neighbors import KNeighborsClassifier
training_accuracy = []
test_accuracy = []

# tryning n_neighbours from 1 to 10

neighbors_settings = range(1,12)
for n_neighbors in neighbors_settings:
    # building the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, Y_train)
    # recording the training set accuracy
    training_accuracy.append(knn.score(X_train, Y_train))
    # testing accuracy
    test_accuracy.append(knn.score(X_test, Y_test))
    
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig("knn_compare_model")
plt.show()


# In[244]:

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,Y_train)

print('Accuracy of k-NN classifier on the training set:{:.2f}'.format(knn.score(X_train, Y_train)))
print('Accuracy of k-NN classifier on the testing set:{:.2f}'.format(knn.score(X_test, Y_test)))


# In[252]:

data.info()


# In[255]:

#LINEAR REGRESSION


# In[256]:

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression


# In[261]:

X = data[['ratingDescription', 'release year', 'user rating score', 'user rating size', 'tsub', 'tpol']].values
Y = data[['f_class']].values


# In[264]:

regr = linear_model.LinearRegression()
regr.fit(X, Y)


# In[271]:

regr.coef_ #slope


# In[272]:

regr.intercept_ #intercept


# In[275]:

print('Slope: %.3f' % regr.coef_[0][0])
print('Intercept: %.3f' % regr.intercept_[0])


# In[290]:

comm = "Parental guidance suggested. May not be suitable for all children."
a = TextBlob(comm)
a_sub = a.sentiment.subjectivity
a_pol = a.sentiment.polarity
print(a_sub)
print(a_pol)


# In[295]:

x = [[70,2016,98,80,a_sub,a_pol]]  


# In[296]:

class_std = regr.predict(x)
print("CLASS: %.3f" %class_std)


# In[ ]:



